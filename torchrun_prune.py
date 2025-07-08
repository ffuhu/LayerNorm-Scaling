import os
os.environ['NORM_TYPE'] = 'LNS'
# --model_config configs/llama_130m.json --lr 1e-4 --batch_size 32 --total_batch_size 64 --num_training_steps 160000 --warmup_steps 2000 --dtype bfloat16 --eval_every 1000 --save_every 1000 --optimizer adamw --beta1 0.98 --weight_decay 0.1 --grad_clipping 0.0 --run_name ew_130m_save0-5-11_ --save_dir logs --layers_to_save layers.0 layers.5 layers.11 --save_every_N_steps 10

import time
import random
import argparse
import numpy as np

import torch
import torch.utils.data


from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM


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
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=1e-8)

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

    # pruning
    parser.add_argument("--sparsity", type=float, default=None)

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
# def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
def evaluate_model(model, preprocess_batched, pad_idx, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True)  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    # if not args.single_gpu:
    #     val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

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

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() # * world_size

    total_loss = total_loss / total_batches

    # # Gather losses across all GPUs
    # gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    # dist.all_gather(gathered_losses, total_loss)
    # total_loss = sum([t.item() for t in gathered_losses]) / world_size
    perplexity = np.exp(total_loss)

    return total_loss, evaluated_on_tokens, perplexity


def get_pruned_names(model):
    masked_names = []
    for name, tensor in model.named_parameters():
        if len(tensor.size()) == 4 or len(tensor.size()) == 2:
            masked_names.append(name)

    # if args.rm_first:
    #     for name, tensor in model.named_parameters():
    #         if 'conv.weight' in name or 'feature.0.weight' in name:
    #             masked_names.pop(name)
    #             print(f"pop out {name}")

    return masked_names


def check_sparsity(model):
    masked_names = get_pruned_names(model)
    total_params = 0
    total_pruned = 0
    for name, tensor in model.named_parameters():
        if name in masked_names:
            layer_total_params = tensor.numel()
            layer_total_pruned = torch.sum(tensor == 0).item()
            total_params += layer_total_params
            total_pruned += layer_total_pruned
            # Calculate sparsity
            sparsity = layer_total_pruned / layer_total_params
            print(
                f"Layer: {name}, Total params: {layer_total_params}, Total pruned: {layer_total_pruned}, Sparsity: {sparsity:.4f}")

    sparsity = total_pruned / total_params
    print(f"Total params: {total_params}, Total pruned: {total_pruned}, Sparsity: {sparsity:.4f}")
    return sparsity


def pruner(model, sparsity):
    pruned_layers = get_pruned_names(model)

    for name, tensor in model.named_parameters():
        if name in pruned_layers:
            # prune tensor to desired sparsity using magnitude-based pruning
            params_to_keep = int(tensor.numel() * (1 - sparsity))
            mask = torch.zeros_like(tensor)

            # rank the weights by their absolute values
            w = tensor.clone().flatten()
            _, indices = torch.topk(w.abs(), params_to_keep, largest=True, sorted=False)
            mask.view(-1)[indices] = 1.0
            tensor.data.mul_(mask)

    check_sparsity(model)
    return model


def main(args):

    assert args.sparsity is not None, "Must specify the sparsity"

    # set saving dir
    args.save_dir = os.path.join(args.save_dir, f"{args.run_name}_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    # global_rank = int(os.environ['RANK'])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # torch.cuda.set_device(local_rank)
    #
    # logger.add(os.path.join(args.save_dir, 'log.txt'))
    #
    # logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")
    #
    # dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    #
    # logger.info("Process group initialized")
    # device = f"cuda:{local_rank}"
    device = f"cuda:0"
    global_rank = 0

    # if args.total_batch_size is not None:
    #     if args.gradient_accumulation is None:
    #         assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
    #         args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
    #         assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"
    #
    # assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
    #     "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # # turn off logger
    # if global_rank != 0: logger.remove()

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting pruning with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 32

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    # if not args.single_gpu:
    #     data = datasets.distributed.split_dataset_by_node(
    #         data, rank=global_rank, world_size=world_size,
    #     )

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
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # if not args.single_gpu:
    #     model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[local_rank],
    #         output_device=local_rank,
    #         broadcast_buffers=False,
    #     )

    pad_idx = tokenizer.pad_token_id


    # Load the model
    checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    logger.info(f"Model successfully loaded (strict=True policy)")
    model.eval()

    # Evaluation before pruning
    logger.info("Running evaluation before pruning")

    total_loss, evaluated_on_tokens, perplexity = evaluate_model(
        # model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
        model, preprocess_batched, pad_idx, device, args.batch_size
    )

    logger.info(f"Eval loss before pruning: {total_loss:.4f}")

    # ##############################
    # PRUNING
    # ##############################
    logger.info(f"Pruning starts")

    model = pruner(model, args.sparsity)

    logger.info("Training finished")

    current_model_directory = f"{args.save_dir}/model_pruned_sparsity_{args.sparsity_level}"
    if global_rank == 0: # and not os.path.exists(current_model_directory):
        logger.info(f"Saving model to {current_model_directory}, sparsity {args.sparsity_level}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

    # Final evaluation
    logger.info("Running final evaluation")

    total_loss, evaluated_on_tokens, perplexity = evaluate_model(
        # model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
        model, preprocess_batched, pad_idx, device, args.batch_size
    )

    logger.info(f"Final eval loss: {total_loss:.4f}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
