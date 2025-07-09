# copy dependencies from transformers/optimization.py
import os
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
import numpy as np
from transformers.utils.versions import require_version

import sys
import h5py

import torch.optim as optim
from tqdm import tqdm

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
            name=None,
            log_folder=None,
            save_every_N_steps=10,
            layers_to_save=None,
            logger=None,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

        # for gradient and weight saving
        self.grad_dict = {}
        self.p_dict = {}
        self.lr_dict = {}
        self.step_dict = {}
        self.partial_saved_steps = 0

        self.moment_dict = {}
        self.name = name
        self.moment_second_dict = {}
        self.log_folder = log_folder
        self.save_every_N_steps = save_every_N_steps
        self.layers_to_save = layers_to_save
        self.logger = logger

        # saving schedule for weights and updates
        self.saving_schedule = {
            0: 1_000,  # 20
            20_000: 2_000,  # 10
            40_000: 4_000,  # 10
            80_000: 10_000,  # 4
            120_000: 4_000,  # 10
            140_000: 1_000,  # 20
        }
        # # FOR TESTING ONLY!!!:
        # self.saving_schedule = {
        #     0: 2,   # 20
        #     100: 10,  # 20
        # }
        # print('\n\n\nWARNING: TESTING WITH SAVING SCHEDULE!!!\n\n\n')

    def should_save_now(self, step):
        saving_stages = list(self.saving_schedule.keys())
        stage_id = torch.nonzero(step > torch.tensor(saving_stages))
        saving_stage_id = stage_id[-1].item() if any(stage_id) else 0
        save_every = self.saving_schedule[saving_stages[saving_stage_id]]
        save_now = step % save_every == 0
        return save_now

    def should_save_weights_for_layer(self, p_name):
        if self.layers_to_save:
            for layer_name in self.layers_to_save:
                if layer_name in p_name:
                    return True
        return False

    def save(self, update_step):

        if self.should_save_now(update_step):
            self.partial_saved_steps += 1

        # for gradient saving
        if self.partial_saved_steps % self.save_every_N_steps == 0 and self.partial_saved_steps > 0:

            optim_name = self.__class__.__name__
            gradient_path = os.path.join(self.log_folder, f"{self.name}_{optim_name}_weights_and_updates.h5")

            # Open or create an HDF5 file
            with h5py.File(gradient_path, 'a') as f:  # 'a' mode allows appending data
                pbar = tqdm(self.grad_dict.keys(), desc='Saving weights and updates')
                for layer_name in pbar:
                    layer_shape = self.grad_dict[layer_name].shape
                    layer_size = sys.getsizeof(self.grad_dict[layer_name]) / 1024 ** 2
                    pbar.set_description(f"Saving gradients for {layer_name} ({layer_size:.2f} MB)")
                    # Create a dataset to store the gradients of each layer
                    if f"{layer_name}_g" not in f:
                        # f.create_dataset(layer_name, data=gradient, compression="gzip", chunks=True)
                        dset_g = f.create_dataset(
                            layer_name + '_g',
                            shape=(0, *layer_shape[-2:]),  # Initial shape
                            maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                            dtype='float16',
                            compression="gzip"  # Optional compression
                        )
                        dset_p = f.create_dataset(
                            layer_name + '_p',
                            shape=(0, *layer_shape[-2:]),  # Initial shape
                            maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                            dtype='float16',
                            compression="gzip"  # Optional compression
                        )
                        dset_lr = f.create_dataset(
                            layer_name + '_lr',
                            shape=(0, 1),  # Initial shape
                            maxshape=(None, 1),  # Allow expansion along axis 0
                            dtype='float16',
                            compression="gzip"  # Optional compression
                        )
                        dset_step = f.create_dataset(
                            layer_name + '_step',
                            shape=(0, 1),  # Initial shape
                            maxshape=(None, 1),  # Allow expansion along axis 0
                            dtype='float32',
                            compression="gzip"  # Optional compression
                        )
                    else:
                        dset_g = f[layer_name + '_g']
                        dset_p = f[layer_name + '_p']
                        dset_lr = f[layer_name + '_lr']
                        dset_step = f[layer_name + '_step']

                    # Resize the dataset to accommodate new data
                    current_size = dset_g.shape[0]
                    new_size = current_size + layer_shape[0]
                    dset_g.resize(new_size, axis=0)
                    dset_p.resize(new_size, axis=0)
                    dset_lr.resize(new_size, axis=0)
                    dset_step.resize(new_size, axis=0)

                    # Write new data at the end of the dataset
                    dset_g[current_size:new_size] = self.grad_dict[layer_name]
                    dset_p[current_size:new_size] = self.p_dict[layer_name]
                    dset_lr[current_size:new_size] = self.lr_dict[layer_name]
                    dset_step[current_size:new_size] = self.step_dict[layer_name]

            msg = f"[STEP: {update_step}] Weights and updates saved at: {gradient_path}"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

            self.grad_dict = {}
            self.p_dict = {}
            self.lr_dict = {}
            self.step_dict = {}
            self.partial_saved_steps = 0

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p_name, p in zip(group["param_names"], group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                state = self.state[id(p)]

                if "step" not in state:
                    state["step"] = 0

                if 'dim' not in group:
                    group['dim'] = 2
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                state["step"] += 1
                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # save the gradient update and the weights
                if self.should_save_weights_for_layer(p_name) and self.should_save_now(state["step"] - 1):
                    if p_name not in self.grad_dict.keys():
                        # if state["step"] - 1 == 0:
                        #     optim_name = self.__class__.__name__
                        #     print(f"[{optim_name}] Save gradients for layer:\t{p_name}\t{grad.shape}")

                        self.grad_dict[p_name] = np.zeros((self.save_every_N_steps, *grad.shape), dtype=np.float16)
                        self.p_dict[p_name] = np.zeros((self.save_every_N_steps, *p.data.shape), dtype=np.float16)
                        self.lr_dict[p_name] = np.zeros((self.save_every_N_steps, 1), dtype=np.float16)
                        self.step_dict[p_name] = np.zeros((self.save_every_N_steps, 1), dtype=np.float32)


                    self.grad_dict[p_name][self.partial_saved_steps] = norm_grad.detach().cpu().float().numpy()
                    self.p_dict[p_name][self.partial_saved_steps] = p.data.detach().cpu().float().numpy()
                    self.lr_dict[p_name][self.partial_saved_steps] = step_size
                    self.step_dict[p_name][self.partial_saved_steps] = state["step"] - 1

                    optim_name = self.__class__.__name__
                    step = state["step"] - 1
                    msg = f"[{optim_name}, step={step}] Added update, weight, lr and step for layer:\t{p_name}\t{grad.shape}"
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))


        return loss
