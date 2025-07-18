import os
os.environ['NORM_TYPE'] = 'LNS'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --model_config configs/llama_130m.json --lr 1e-4 --batch_size 32 --total_batch_size 64 --num_training_steps 160000 --warmup_steps 2000 --dtype bfloat16 --eval_every 1000 --save_every 1000 --optimizer adamw --beta1 0.98 --weight_decay 0.1 --grad_clipping 0.0 --run_name ew_130m_save0-5-11_ --save_dir logs --layers_to_save layers.0 layers.5 layers.11 --save_every_N_steps 10

import re
import time
import json
import h5py
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from peft_pretraining.modeling_llama import LlamaForCausalLM

import matplotlib.pyplot as plt
import seaborn as sns


def get_pruned_names(model):
    masked_names = []
    for name, tensor in model.named_parameters():
        if len(tensor.size()) == 4 or len(tensor.size()) == 2:
            masked_names.append(name)

    return masked_names


def check_effective_weights(model, threshold, layers_to_check=None):
    masked_names = get_pruned_names(model)
    total_params = 0
    total_effective_weights = 0
    effective_weights_per_layer = {}
    pcnt_effective_weights_per_layer = {}
    for name, tensor in model.named_parameters():
        if name in masked_names:
            layer_total_params = tensor.numel()
            layer_total_effective_weights = torch.sum(tensor > threshold).item()
            total_params += layer_total_params
            total_effective_weights += layer_total_effective_weights
            # Calculate effective weights
            pcnt_layer_total_effective_weights = layer_total_effective_weights / layer_total_params
            effective_weights_per_layer[name] = layer_total_effective_weights
            pcnt_effective_weights_per_layer[name] = pcnt_layer_total_effective_weights


    pcnt_effective_weights = total_effective_weights / total_params
    return pcnt_effective_weights, effective_weights_per_layer, pcnt_effective_weights_per_layer


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


def get_pruned_mask(tensor, sparsity):
    # with threshold
    # return (tensor.abs() < threshold)
    # with sparsity level
    params_to_keep = int(tensor.numel() * (1 - sparsity))
    mask = torch.zeros_like(tensor)

    # rank the weights by their absolute values
    w = tensor.clone().flatten()
    _, indices = torch.topk(w.abs(), params_to_keep, largest=True, sorted=False)
    mask.view(-1)[indices] = 1.0
    return mask.to(torch.bool)


def analyze_pruned_gradients(model, gradients_path, layer_name, sparsity, steps_to_analyze, exp_name):
    # get pruned mask
    final_weights = dict(model.named_parameters())[layer_name].detach().cpu()
    pruned_mask = get_pruned_mask(final_weights, sparsity)  # shape: same as weights

    # open gradients file
    with h5py.File(gradients_path, 'r') as f:
        grad_key = f"{layer_name}_g"

        grad_cum_not_pruned = np.zeros((~pruned_mask).sum(), dtype=np.float32)
        grad_cum_pruned = np.zeros(pruned_mask.sum(), dtype=np.float32)
        grad_norms_over_time_not_pruned = np.zeros((len(steps_to_analyze), (~pruned_mask).sum()), dtype=np.float32)
        grad_norms_over_time_pruned = np.zeros((len(steps_to_analyze), pruned_mask.sum()), dtype=np.float32)
        for i_step, step in enumerate(tqdm(steps_to_analyze, desc=f"Computing gradients for layer {layer_name}")):
            # Gradients are usually stored as [step, ...weight_shape...]
            grads = f[grad_key][i_step]  # shape: same as weights
            grads = torch.tensor(grads)
            # pruned weights norms
            pruned_grads = grads[pruned_mask]
            grad_norms_pruned = pruned_grads.abs().numpy()
            grad_norms_over_time_pruned[i_step] = grad_norms_pruned
            # not pruned weights norms
            not_pruned_grads = grads[~pruned_mask]
            grad_norms_not_pruned = not_pruned_grads.abs().numpy()
            grad_norms_over_time_not_pruned[i_step] = grad_norms_not_pruned
            # pruned weights cumulative
            grad_cum_pruned += pruned_grads.numpy()
            # not pruned weights cumulative
            grad_cum_not_pruned += not_pruned_grads.numpy()


    # mean over time - pruned and not pruned
    means_pruned = np.mean(grad_norms_over_time_pruned, axis=1)
    means_not_pruned = np.mean(grad_norms_over_time_not_pruned, axis=1)
    plt.figure()
    plt.plot(steps_to_analyze, means_pruned, 'r')
    plt.plot(steps_to_analyze, means_not_pruned, 'b')
    plt.xlabel('Step')
    plt.ylabel('Mean Gradient Norm')
    plt.title(f'Mean Gradient Norms for Pruned vs Not Pruned Weights: {layer_name}')
    plt.legend()
    # plt.show()
    plt.savefig(exp_name + f'_{layer_name}_gradient_norm_evolution.png')

    # plot grad norms - pruned vs not pruned
    # for i, step in enumerate(steps_to_analyze[::10]):
    for i, step in enumerate([0, len(steps_to_analyze) // 2, -1]):
        step = steps_to_analyze[step]
        # plt.figure()
        # plt.hist(grad_norms_over_time[i], bins=50)
        # plt.xlabel('Gradient Norm')
        # plt.ylabel('Count')
        # plt.title(f'Gradient Norms (pruned) at step {steps_to_analyze[i]}')
        # plt.show()
        plt.figure(figsize=(10, 6))

        # Plotting the first histogram
        plt.hist(
            grad_norms_over_time_pruned[steps_to_analyze.index(step)],
            bins=50,  # Number of bins
            density=True,  # Normalize to form a probability density
            alpha=0.6,  # Transparency
            color="blue",
            label="Gradients of pruned weights",
        )

        # Plotting the second histogram
        plt.hist(
            grad_norms_over_time_not_pruned[steps_to_analyze.index(step)],
            bins=50,
            density=True,
            alpha=0.6,
            color="red",
            label="Gradients of NOT pruned weights",
        )

        # # Plotting the first KDE
        # sns.kdeplot(grad_norms_over_time[steps_to_analyze.index(step)],
        #             fill=True, color="blue", label="Distribution 1", alpha=0.5)
        #
        # # Plotting the second KDE
        # sns.kdeplot(grad_norms_over_time_not_pruned[steps_to_analyze.index(step)],
        #             fill=True, color="red", label="Distribution 2", alpha=0.5)

        plt.title(f'Gradient Norms at step {step} - {layer_name}')
        plt.xlabel("Gradient norm")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        # plt.show()
        plt.savefig(exp_name + f'_{layer_name}_{step}_gradient_dist.png')

    # plot grad cums - pruned vs not pruned
    plt.figure(figsize=(10, 6))

    # Plotting the first histogram
    plt.hist(
        grad_cum_pruned,
        bins=50,  # Number of bins
        density=True,  # Normalize to form a probability density
        alpha=0.6,  # Transparency
        color="blue",
        label="Gradients of pruned weights",
    )

    # Plotting the second histogram
    plt.hist(
        grad_cum_not_pruned,
        bins=50,
        density=True,
        alpha=0.6,
        color="red",
        label="Gradients of NOT pruned weights",
    )

    plt.title(f'Cumulative Gradient - {layer_name}')
    plt.xlabel("Cumulative Gradient")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    # plt.show()
    plt.savefig(exp_name + f'_{layer_name}_cummulative_gradient.png')

    return {'grad_means_pruned': means_pruned,
            'grad_means_not_pruned': means_not_pruned,
            'grad_norms_over_time_pruned': grad_norms_over_time_pruned,
            'grad_norms_over_time_not_pruned': grad_norms_over_time_not_pruned,
            'grad_cum_pruned': grad_cum_pruned,
            'grad_cum_not_pruned': grad_cum_not_pruned,
            }

def compute_weight_distributions(model, checkpoint_path):

    # load the model
    # checkpoint_path = os.path.join(args.ckpts_folder, f"model_{update_step}/pytorch_model.bin")
    # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

    blocks_to_plot = ['layers.0', 'layers.5', 'layers.9']


    # Get all weight parameters
    weight_layers = []
    layer_names = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad and 'bn' not in name and any([block in name for block in blocks_to_plot]):
            weight_layers.append(param.detach().cpu().float().numpy().flatten())
            layer_names.append(name)

    # Calculate grid size
    n_layers = len(weight_layers)
    cols = 9  # Number of columns
    rows = (n_layers + cols - 1) // cols  # Calculate rows needed

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Flatten axes array for easier indexing
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    # Plot each layer's weight distribution
    for i, (weights, name) in enumerate(tqdm(zip(weight_layers, layer_names),
                                        total=len(weight_layers), desc="Computing histograms")):
        ax = axes[i]

        # Create histogram
        weights = weights[~np.isnan(weights)]
        if weights.size == 0:
            print(f"Warning: {name} weights are NaN. Skipping histogram.")
            continue
        ax.hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

        # Formatting
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.9)

        # Add statistics as text
        mean_val = np.mean(weights)
        std_val = np.std(weights)
        ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Add main title AFTER tight_layout with proper positioning
    exp_name = checkpoint_path.split(os.sep)
    exp_name = f"{exp_name[-3]} {exp_name[-3]}"
    fig.suptitle(f'Weight Distribution: {exp_name}', fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.94)
    # plt.show()
    plt.savefig(exp_name + f'_weight_dist_withwd.png')
    plt.close()

    print(f"Total layers with weights: {n_layers}")
    print("Layer names:")
    for i, name in enumerate(layer_names):
        print(f"{i + 1:2d}. {name}")

def run_analysis(model, steps_to_analyze, ckpts_folder, experiment, threshold):

    # experiment = 'ew_130m_save0-5-11__adam_mini_lr0.0001_wd0.1_seed1'
    # print("DEBUG; REMOVE!!! (l.331)")
    print(experiment)

    # ==================================================================================================================
    # compute effective weights
    # ==================================================================================================================

    # to store metrics
    pcnt_effective_weights, effective_weights_per_layer, pcnt_effective_weights_per_layer = {}, {}, {}

    list_of_update_steps = [i * 10_000 for i in range(1, 17)]
    for update_step in list_of_update_steps:

        checkpoint_path = os.path.join(ckpts_folder, experiment, f"model_{update_step}/pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        print(f"Model for update_step {update_step} successfully loaded (strict=True policy)")

        device = f"cuda:0"
        if torch.cuda.is_bf16_supported():
            model = model.to(device=device, dtype=torch.bfloat16)
        else:
            model = model.to(device=device)

        pcnt_effective_weights_model, effective_weights_per_layer_model, pcnt_effective_weights_per_layer_model = (
            check_effective_weights(model, threshold))

        pcnt_effective_weights[update_step] = pcnt_effective_weights_model
        effective_weights_per_layer[update_step] = effective_weights_per_layer_model
        pcnt_effective_weights_per_layer[update_step] = pcnt_effective_weights_per_layer_model

    metrics = {
        'pcnt_effective_weights': pcnt_effective_weights,
        'effective_weights_per_layer': effective_weights_per_layer,
        'pcnt_effective_weights_per_layer': pcnt_effective_weights_per_layer,
    }

    # # Create subplots
    # fig, ax = plt.subplots(figsize=(12, 12))
    #
    # ax.plot(pcnt_effective_weights.keys(), pcnt_effective_weights.values())
    #
    # ax.set_xlabel('Training step')
    # ax.set_ylabel('% of effective weights')
    # ax.grid(True, alpha=0.9)
    #
    # plt.tight_layout()
    #
    # # Add main title AFTER tight_layout with proper positioning
    # exp_name = checkpoint_path.split(os.sep)
    # exp_name = f"{exp_name[-3]}"
    # fig.suptitle(f'{exp_name} - threshold={threshold:.6f}', fontsize=16, fontweight='bold', y=0.98)
    #
    # # Adjust layout to make room for the title
    # plt.subplots_adjust(top=0.94)
    # # plt.show()
    # plt.savefig(exp_name + f'_effective_weights.png')
    # plt.close()
    return metrics


    # # ==================================================================================================================
    # # inspect gradients
    # # ==================================================================================================================
    # # load gradients
    #
    # gradients_path = glob.glob(os.path.join(ckpts_folder, experiment) + '/*.h5')[-1]
    # training_gradients = h5py.File(gradients_path)
    #
    # pruned_layers = get_pruned_names(model)
    # layer_names = list({'_'.join(layer_name.split('_')[:-1]) for layer_name in list(training_gradients.keys())})
    # for layer_name in layer_names:
    #
    #     if layer_name not in pruned_layers: continue
    #
    #     metrics_gradients_layer = analyze_pruned_gradients(
    #         model,
    #         gradients_path,
    #         layer_name,
    #         sparsity=args.sparsity,
    #         steps_to_analyze=steps_to_analyze,
    #         exp_name=exp_name,
    #     )
    #
    #     metrics['metrics_gradients_layer'] = {}
    #     metrics['metrics_gradients_layer'][layer_name] = metrics_gradients_layer
    # print('done')

def main():

    model_config = '../configs/llama_130m.json'
    ckpts_folder = '../logs/'
    seed = 1

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    threshold = 1e-3
    steps_to_analyze = [*list(range(0, 21, 1)),
                        *list(range(21, 41, 2)),
                        *list(range(41, 80, 4)),
                        *list(range(81, 121, 10)),
                        *list(range(121, 140, 4)),
                        *list(range(141,161, 1))]

    model_config = AutoConfig.from_pretrained(model_config)
    model = LlamaForCausalLM(model_config)

    # gather all experiment checkpoints
    ckpts = glob.glob(ckpts_folder + 'ew*/**/*.txt', recursive=True)
    experiments = {ckpt.split(os.sep)[2] for ckpt in ckpts}

    # print("DEBUG!!!!! l439")
    # experiments = ['ew_130m_save0-5-11__adam_mini_lr0.0001_wd0.1_seed1']
    # ckpts_folder = '../logs_server/'

    metrics = {}
    for experiment in experiments:
        try:
            metrics[experiment] = run_analysis(model, steps_to_analyze, ckpts_folder, experiment, threshold)
        except:
            print(f"Experiment {experiment} not found")
            continue

    with open('effective_weights.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # make the plot
    opts = ['adam_mini', 'adamw', 'sgd', 'muon']
    metrics_opts = {}
    for opt in opts:
        metrics_opts[opt] = {k: v for k, v in metrics.items() if opt in k}

    # Create subplots
    n_plots = 1
    nrows = 1
    ncols = len(opts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]

    # Plot each experiment effective weights
    for i, opt in enumerate(opts):
        ax = axes[i]


        lrs_ew = [(float(re.search(r'_lr([0-9.]+)', k).group(1)), v['pcnt_effective_weights'])
                  for k, v in metrics_opts[opt].items()]

        for i_lr, (lr, step_ew) in enumerate(lrs_ew):
            steps = [f"{v//1_000}k" for v in list(step_ew.keys())]
            ews = list(step_ew.values())
            print(opt, lr, ews)

            ax.plot(steps, ews, label=f"{lr}", color=colors[i_lr])

        # Formatting
        ax.set_title(f"{opt}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Update step')
        ax.grid(True, alpha=0.9)
        ax.set_ylim(0, 1)
        plt.legend()

        n_plots += 1

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    # plt.tight_layout()
    #plt.legend()

    # Add main title AFTER tight_layout with proper positioning
    fig.suptitle(f'Effective weights', fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout to make room for the title
    # plt.subplots_adjust(top=0.94)
    # plt.show()
    plt.savefig(f'effective_weights.png')
    plt.close()





if __name__ == "__main__":
    print("Starting script")
    main()
