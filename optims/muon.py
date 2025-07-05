import torch
import torch.distributed as dist

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                        dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                            dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()],
                                    params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups,
                 name=None,
                 log_folder=None,
                 save_every_N_steps=10,
                 layers_to_save=None,
                 ):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

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
        # self.saving_schedule = {
        #     0: 1_000,  # 20
        #     20_000: 2_000,  # 10
        #     40_000: 4_000,  # 10
        #     80_000: 10_000,  # 4
        #     120_000: 4_000,  # 10
        #     140_000: 1_000,  # 20
        # }
        # FOR TESTING ONLY!!!:
        self.saving_schedule = {
            0: 2,  # 20
            100: 10,  # 20
        }

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

    def add_weights_and_updates_to_save_dict(self, name, step, p, update, lr):
        # save the gradient update and the weights
        if self.should_save_weights_for_layer(name) and self.should_save_now(step):
            if name not in self.grad_dict.keys():
                if step == 0:
                    optim_name = self.__class__.__name__
                    print(f"[{optim_name}] Save gradients for layer:\t{name}\t{update.shape}")

                self.grad_dict[name] = np.zeros((self.save_every_N_steps, *update.shape), dtype=np.float16)
                self.p_dict[name] = np.zeros((self.save_every_N_steps, *p.data.shape), dtype=np.float16)
                self.lr_dict[name] = np.zeros((self.save_every_N_steps, 1), dtype=np.float16)
                self.step_dict[name] = np.zeros((self.save_every_N_steps, 1), dtype=np.uint16)

            self.grad_dict[name][self.partial_saved_steps] = update.detach().cpu().float().numpy()
            self.p_dict[name][self.partial_saved_steps] = p.data.detach().cpu().float().numpy()
            self.lr_dict[name][self.partial_saved_steps] = lr
            self.step_dict[name][self.partial_saved_steps] = step #state["step"] - 1

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # for p in group["params"]:
                for p_name, p in zip(group["param_names"], group["params"]):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])

                    state["step"] += 1

                    # add weights and updates to be saved
                    self.add_weights_and_updates_to_save_dict(p_name, state["step"] - 1, p, update, group['lr'])

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # for p in group["params"]:
                for p_name, p in zip(group["param_names"], group["params"]):
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])

                    state["step"] += 1

                    # add weights and updates to be saved
                    self.add_weights_and_updates_to_save_dict(p_name, state["step"] - 1, p, update, group['lr'])

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])


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