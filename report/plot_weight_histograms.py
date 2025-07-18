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


def compute_weight_distributions(model, checkpoints_folder, exp_names, optimizer_name, update_step="160001"):

    blocks_to_plot = ['layers.0', 'layers.5', 'layers.9']
    fig_created = False
    # all_weights = {}
    # normalizers = {}
    # for i_exp, exp_name in enumerate(tqdm(exp_names, desc="Computing max for normalization")):
    #
    #     if i_exp == 0:
    #         normalizers[exp_name] = 0.
    #
    #     # load the model
    #     checkpoint_path = os.path.join(checkpoints_folder, f"{exp_name}/model_160001/pytorch_model.bin")
    #     model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    #     for name, param in model.named_parameters():
    #         if 'weight' in name and param.requires_grad and 'bn' not in name and any(
    #                 [block in name for block in blocks_to_plot]):
    #             normalizers[exp_name] = np.max([normalizers[exp_name], param.detach().cpu().float().numpy().max()])
    #
    #     # Get all weight parameters
    #     all_weights[exp_name] = {}
    #     for name, param in model.named_parameters():
    #         if 'weight' in name and param.requires_grad and 'bn' not in name and any(
    #                 [block in name for block in blocks_to_plot]):
    #             all_weights[exp_name][name] = param.detach().cpu().float().numpy().flatten() / normalizers[exp_name]

    for i_exp, exp_name in enumerate(exp_names):

        lr = re.search(r'_lr([0-9.]+)', exp_name).group(1)

        # load the model
        try:
            checkpoint_path = os.path.join(checkpoints_folder, f"{exp_name}/model_{update_step}/pytorch_model.bin")
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        except FileNotFoundError as e:
            print(f"Could not find checkpoint {checkpoint_path}")
            continue

        # Get all weight parameters
        weight_layers = []
        layer_names = []

        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad and 'bn' not in name and any([block in name for block in blocks_to_plot]):
                p = param.detach().cpu().float().numpy().flatten()
                weight_layers.append(p / np.abs(p).max())  # normalize
                layer_names.append(name)

        # plot each lr for a given optim in the same plot
        colors = "rgbcmyk"
        if not fig_created:
            fig_created = True
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
                                                 total=len(weight_layers),
                                                 desc=f"Computing histograms for {optimizer_name} and {lr}")):
            ax = axes[i]

            # Create histogram
            weights = weights[~np.isnan(weights)]
            if weights.size == 0:
                print(f"Warning: {name} weights are NaN. Skipping histogram.")
                continue
            # ax.hist(weights, bins=50, alpha=0.3, color=colors[i_exp], edgecolor='black', linewidth=0.5, label=lr)
            ax.hist(weights, bins=50, alpha=0.9, color=colors[i_exp], edgecolor='black', linewidth=0.5, label=lr)

            # Formatting
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.9)
            ax.set_yscale('log')

            # # Add statistics as text
            # mean_val = np.mean(weights)
            # std_val = np.std(weights)
            # ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
            #         transform=ax.transAxes, verticalalignment='top',
            #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.legend()

    # Add main title AFTER tight_layout with proper positioning
    plt.suptitle(f'Weight Distribution: {optimizer_name} - {update_step}', fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.94)
    # plt.show()
    plt.savefig(optimizer_name + f'_weight_dist_withwd_{update_step}.png')
    plt.close()



def main():

    model_config = '../configs/llama_130m.json'
    ckpts_folder = '../logs_server/'
    seed = 1
    # update_step = "160001"
    # update_step = "10000"
    update_step = "80000"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_config = AutoConfig.from_pretrained(model_config)
    model = LlamaForCausalLM(model_config)

    # gather all experiment checkpoints
    ckpts = glob.glob(ckpts_folder + 'ew*/**/*.txt', recursive=True)
    experiments = {ckpt.split(os.sep)[2] for ckpt in ckpts}
    optimizers = {re.search(r'__([a-zA-Z_]+)_lr', ckpt).group(1) for ckpt in ckpts}

    for optimizer in optimizers:

        exp_names = sorted([exp for exp in experiments if optimizer in exp], reverse=True)
        compute_weight_distributions(model, ckpts_folder, exp_names, optimizer, update_step)





if __name__ == "__main__":
    print("Starting script")
    # args = parse_args(None)
    main()
