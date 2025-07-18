import os
import sys
# --model_config configs/llama_130m.json --lr 1e-4 --batch_size 32 --total_batch_size 64 --num_training_steps 160000 --warmup_steps 2000 --dtype bfloat16 --eval_every 1000 --save_every 1000 --optimizer adamw --beta1 0.98 --weight_decay 0.1 --grad_clipping 0.0 --run_name ew_130m_save0-5-11_ --save_dir logs --layers_to_save layers.0 layers.5 layers.11 --save_every_N_steps 10

import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

from optims.sgd import SGD
from optims.adamw import AdamW
from optims.adam_mini import Adam_mini
from optims.muon import SingleDeviceMuonWithAuxAdam
from optims.soap import SOAP
# from torch.optim.sgd import SGD

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb

import matplotlib.pyplot as plt

#transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default="default")
    # optim parameters
    parser.add_argument("--beta1", type=float, default=0.98)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--momentum", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--precondition_frequency", type=int, default=50)

    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    # to save weights and grads
    parser.add_argument("--save_every_N_steps", type=int, default=None)
    parser.add_argument("--layers_to_save", type=str, nargs='+', default=[])

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True)  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size
    perplexity = np.exp(total_loss)

    return total_loss, evaluated_on_tokens, perplexity


def main(args):

    # set saving dir
    args.save_dir = os.path.join(args.save_dir, f"{args.run_name}_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.add(os.path.join(args.save_dir, 'log.txt'))

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="cod", name=args.run_name)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 32

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")

        from safetensors.torch import load_file
        state_dict = load_file(f"{args.continue_from}/model.safetensors")
        model.load_state_dict(state_dict)

        # checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    if 'galore' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params},
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap,
                         'scale': args.galore_scale, 'proj_type': args.proj_type}]

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    layer_wise_flag = False
    # if args.optimizer.lower() == "adam":
    #     optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # else:
    #     raise ValueError(f"Optimizer {args.optimizer} not supported")
    if args.optimizer.lower() == "sgd":
        optimizer = SGD(model.named_parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True,
                        save_every_N_steps=args.save_every_N_steps,
                        layers_to_save=args.layers_to_save,
                        log_folder=args.save_dir,
                        logger=logger,
        )
    elif args.optimizer.lower() == 'adamw':
        # optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay,
        #                               betas=(args.beta1, args.beta2), eps=args.eps)
        optimizer = AdamW(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay,
                          betas=(args.beta1, args.beta2), eps=args.eps,
                          save_every_N_steps=args.save_every_N_steps,
                          layers_to_save=args.layers_to_save,
                          log_folder=args.save_dir,
                          logger=logger,
                          )
    elif args.optimizer.lower() == 'adam_mini':
        optimizer = Adam_mini(
            named_parameters=model.named_parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            dim=model_config.hidden_size,
            n_heads=model_config.num_attention_heads,
            n_kv_heads=model_config.num_key_value_heads,  # default to be none
            save_every_N_steps=args.save_every_N_steps,
            layers_to_save=args.layers_to_save,
            log_folder=args.save_dir,
            logger=logger,
        )

    elif args.optimizer.lower() == 'muon':

        hidden_weights = [p for p in model.model.layers.parameters() if p.ndim >= 2]
        hidden_weights_names = [f"layers.{n}" for n, p in model.model.layers.named_parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.model.layers.parameters() if p.ndim < 2]
        hidden_gains_biases_names = [f"layers.{n}" for n, p in model.model.layers.named_parameters() if p.ndim < 2]
        nonhidden_params = [*model.lm_head.parameters(), *model.model.embed_tokens.parameters()]
        nonhidden_params_names = [*[f"lm_head.{n}" for n, p in model.lm_head.named_parameters()],
                                  *[f"embed_tokens.{n}" for n, p in model.model.embed_tokens.named_parameters()]]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=args.lr, weight_decay=args.weight_decay,
                 param_names=hidden_weights_names),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=1e-3, betas=(0.95, 0.95), weight_decay=0.1,
                 param_names=hidden_gains_biases_names + nonhidden_params_names),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups,
                                                save_every_N_steps=args.save_every_N_steps,
                                                layers_to_save=args.layers_to_save,
                                                log_folder=args.save_dir,
                                                logger=logger,
                                                )
    elif args.optimizer.lower() == 'soap':
        optimizer = SOAP(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay,
                         precondition_frequency=args.precondition_frequency,
                         betas=(args.beta1, args.beta2), eps=args.eps,
                         save_every_N_steps=args.save_every_N_steps,
                         layers_to_save=args.layers_to_save,
                         log_folder=args.save_dir,
                         logger=logger,
                         )

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    logger.info(f"Optimizer {args.optimizer} configured.")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    logger.info(f"Training starts")
    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1
        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps ({args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        if torch.isnan(loss):
            logger.info('Loss is nan. Aborting execution...')
            sys.exit(1)

        # The below code is only executed during the update step

        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0:
            pbar.update(1)
            # if update_step % 1_000 == 0:
            #     logger.info(f"Update step: {update_step}.")

        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # to save weights and updates every optim.save_every_N_steps according to optim.saving_schedule
            if global_rank == 0:
                optimizer.save(update_step=update_step)

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            tokenizer.save_pretrained(current_model_directory)
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(current_model_directory, max_shard_size='100GB')
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens, perplexity = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    "perplexity": perplexity,
                },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            pass
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
            },
                step=global_step,
            )
            if update_step % 1_000 == 0:
                logger.info(f"Update step: {update_step}. Loss: {loss.item():.4f}. Tokens seen: {tokens_seen}.")
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0: # and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc;
    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens, perplexity = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            "perplexity": perplexity,
        },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
