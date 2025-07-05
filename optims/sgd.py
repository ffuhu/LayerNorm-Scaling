import torch
from torch.optim.optimizer import Optimizer

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
    """

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, maximize=False,
                 name=None,
                 log_folder=None,
                 save_every_N_steps=10,
                 layers_to_save=None,
                 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov,
                       maximize=maximize)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

        # for gradient and weight saving
        self.grad_dict = {}
        self.p_dict = {}
        self.lr_dict = {}
        self.step_dict = {}
        self.partial_saved_steps = 0

        self.name = name
        self.log_folder = log_folder
        self.save_every_N_steps = save_every_N_steps
        self.layers_to_save = layers_to_save

        # THIS IS THE GOOD ONE:
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
        #     0: 2,  # 20
        #     100: 10,  # 20
        # }

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

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']

            # for p in group['params']:
            for p_name, p in zip(group["param_names"], group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                d_p = p.grad
                if maximize:
                    d_p = -d_p

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    # param_state = self.state[p]
                    # if len(param_state) == 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    buf = state["momentum_buffer"]

                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                state["step"] += 1

                # save the gradient update and the weights
                if self.should_save_weights_for_layer(p_name) and self.should_save_now(state["step"] - 1):
                    if p_name not in self.grad_dict.keys():
                        if state["step"] - 1 == 0:
                            optim_name = self.__class__.__name__
                            print(f"[{optim_name}] Save gradients for layer:\t{p_name}\t{d_p.shape}")

                        self.grad_dict[p_name] = np.zeros((self.save_every_N_steps, *d_p.shape),
                                                          dtype=np.float16)
                        self.p_dict[p_name] = np.zeros((self.save_every_N_steps, *p.data.shape),
                                                       dtype=np.float16)
                        self.lr_dict[p_name] = np.zeros((self.save_every_N_steps, 1), dtype=np.float16)
                        self.step_dict[p_name] = np.zeros((self.save_every_N_steps, 1), dtype=np.uint16)

                    self.grad_dict[p_name][self.partial_saved_steps] = d_p.detach().cpu().float().numpy()
                    self.p_dict[p_name][self.partial_saved_steps] = p.data.detach().cpu().float().numpy()
                    self.lr_dict[p_name][self.partial_saved_steps] = group["lr"]
                    self.step_dict[p_name][self.partial_saved_steps] = state["step"] - 1

                p.add_(d_p, alpha=-group['lr'])

        if self.should_save_now(state["step"] - 1):
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
                            dtype='uint16',
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

            print("Saved at", gradient_path)
            self.grad_dict = {}
            self.p_dict = {}
            self.lr_dict = {}
            self.step_dict = {}
            self.partial_saved_steps = 0

        return loss