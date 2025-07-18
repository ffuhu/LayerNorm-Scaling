from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import math

def add_sparse_args(parser):
    # hyperparameters for Zero-Cost Neuroregeneration
    parser.add_argument('--growth', type=str, default='gradient', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / death rate for Zero-Cost Neuroregeneration.')
    parser.add_argument('--pruning-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--update-frequency', type=int, default=100, metavar='N', help='how many iterations to train between mask update')
    parser.add_argument('--sparse-init', type=str, default='ERK, uniform distributions for sparse training, global pruning and uniform pruning for pruning', help='sparse initialization')
    # hyperparameters for gradually pruning
    parser.add_argument('--method', type=str, default='GraNet', help='method name: DST, GraNet, GraNet_uniform, GMP, GMO_uniform, LIFT_uniform')
    parser.add_argument('--init-density', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--final-density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--init-prune-epoch', type=int, default=0, help='The pruning rate / death rate.')
    parser.add_argument('--final-prune-epoch', type=int, default=110, help='The density of the overall sparse network.')
    parser.add_argument('--rm-first', action='store_true', help='Keep the first layer dense.')


class CosineDecay(object):
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, prune_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, prune_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return prune_rate*self.factor
        else:
            return prune_rate



class Masking(object):
    def __init__(self, optimizer, prune_rate=0.3, growth_death_ratio=1.0, prune_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, args=None, train_loader=None, device=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.loader = train_loader
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.sparse_init = args.sparse_init


        self.masks = {}
        self.final_masks = {}
        self.grads = {}
        self.nonzero_masks = {}
        self.scores = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer if isinstance(optimizer, list) else [optimizer]

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.total_params = 0
        self.fc_params = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0

        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency


    def init(self, mode='ER', density=0.05, erk_power_scale=1.0, grad_dict=None):
        if self.args.method == 'GMP':
            print('initialized with GMP, ones')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()
        elif self.sparse_init == 'prune_uniform':
            # used for pruning stabability test
            print('initialized by prune_uniform')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (weight!=0).cuda()
                    num_zeros = (weight==0).sum().item()
                    num_remove = (self.args.pruning_rate) * self.masks[name].sum().item()
                    k = math.ceil(num_zeros + num_remove)
                    if num_remove == 0.0: return weight.data != 0.0
                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    self.masks[name].data.view(-1)[idx[:k]] = 0.0
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()

        elif self.sparse_init == 'prune_global':
            # used for pruning stabability test
            print('initialized by prune_global')
            self.baseline_nonzero = 0
            total_num_nonzoros = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (weight!=0).cuda()
                    self.name2nonzeros[name] = (weight!=0).sum().item()
                    total_num_nonzoros += self.name2nonzeros[name]

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(total_num_nonzoros * (1 - self.args.pruning_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
            self.apply_mask()

        elif self.sparse_init == 'prune_and_grow_uniform':
            # used for pruning stabability test
            print('initialized by pruning and growing uniformly')

            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    # prune
                    self.masks[name] = (weight!=0).cuda()
                    num_zeros = (weight==0).sum().item()
                    num_remove = (self.args.pruning_rate) * self.masks[name].sum().item()
                    k = math.ceil(num_zeros + num_remove)
                    if num_remove == 0.0: return weight.data != 0.0
                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    self.masks[name].data.view(-1)[idx[:k]] = 0.0
                    total_regrowth = (self.masks[name]==0).sum().item() - num_zeros

                    # set the pruned weights to zero
                    weight.data = weight.data * self.masks[name]
                    for optim in self.optimizer:
                        if weight in optim.state:
                            if 'momentum_buffer' in optim.state[weight]:
                                optim.state[weight]['momentum_buffer'] = optim.state[weight]['momentum_buffer'] * self.masks[name]

                    # grow
                    grad = grad_dict[name]
                    grad = grad * (self.masks[name] == 0).float()

                    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                    self.masks[name].data.view(-1)[idx[:total_regrowth]] = 1.0
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()

        elif self.sparse_init == 'prune_and_grow_global':
            # used for pruning stabability test
            print('initialized by pruning and growing globally')
            self.baseline_nonzero = 0
            total_num_nonzoros = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (weight!=0).cuda()
                    self.name2nonzeros[name] = (weight!=0).sum().item()
                    total_num_nonzoros += self.name2nonzeros[name]

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(total_num_nonzoros * (1 - self.args.pruning_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

                    # set the pruned weights to zero
                    weight.data = weight.data * self.masks[name]
                    for optim in self.optimizer:
                        if weight in optim.state:
                            if 'momentum_buffer' in optim.state[weight]:
                                optim.state[weight]['momentum_buffer'] = optim.state[weight]['momentum_buffer'] * self.masks[name]

            ### grow
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    total_regrowth = self.name2nonzeros[name] - (self.masks[name]!=0).sum().item()
                    grad = grad_dict[name]
                    grad = grad * (self.masks[name] == 0).float()

                    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                    self.masks[name].data.view(-1)[idx[:total_regrowth]] = 1.0
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()

        elif self.sparse_init == 'uniform':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()  # lsw
                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += weight.numel() * density
            self.apply_mask()

        elif self.sparse_init == 'ERK':
            print('initialize by ERK')
            for name, weight in self.masks.items():
                self.total_params += weight.numel()
                if 'classifier' in name:
                    self.fc_params = weight.numel()
            is_epsilon_valid = False
            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / self.total_params}")

        self.apply_mask()


        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()
        print('Total parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))

    def step(self):
        for opt in self.optimizer:
            opt.step()
        self.apply_mask()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps += 1

        if self.prune_every_k_steps is not None:
                if self.args.method == 'GraNet':
                    if self.steps >= (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                        self.pruning(self.steps)
                        self.truncate_weights(self.steps)
                        self.print_nonzero_counts()
                elif self.args.method == 'GraNet_uniform':
                    if self.steps >= (self.args.init_prune_epoch * len(
                            self.loader) * self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                        self.pruning_uniform(self.steps)
                        self.truncate_weights(self.steps)
                        self.print_nonzero_counts()
                    # _, _ = self.fired_masks_update()
                elif self.args.method == 'DST':
                    if self.steps % self.prune_every_k_steps == 0:
                        self.truncate_weights()
                        self.print_nonzero_counts()
                elif self.args.method == 'GMP':
                    if self.steps >= (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                        self.pruning(self.steps)
                elif self.args.method == 'GMP_uniform':
                    if self.steps >= (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                        self.pruning_uniform(self.steps)
                elif self.args.method == 'LIFT_uniform':
                    if self.steps >= (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                        self.pruning_lift_uniform(self.steps)

    def pruning(self, step):
        # prune_rate = 1 - self.args.final_density - self.args.init_density
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int((self.args.final_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps)
        ini_iter = int((self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')


        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter - 1:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) > acceptable_score).float() # must be > to prevent acceptable_score is zero, leading to dense tensors

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format(
                (total_size-sparse_size) / total_size))

    def pruning_lift_uniform(self, step, conv_norm=0.5):
        # prune_rate = 1 - self.args.final_density - self.args.init_density
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = (self.args.final_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps
        ini_iter = (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')

        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)
            # keep the density of the last layer as 0.2 if spasity is larger then 0.8
            if curr_prune_rate >= 0.8:
                curr_prune_rate = 1 - (self.total_params * (1-curr_prune_rate) - 0.2 * self.fc_params)/(self.total_params-self.fc_params)

                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue

                        prev_mask = self.masks[name].view(weight.shape[0], -1)
                        assert len(weight.shape) == 4 or len(weight.shape) == 2, 'weight shape should be 4 or 2'
                        filter_rank = min(weight.shape[0] - 1, weight.shape[1] - 1, self.args.filter_rank)
                        if len(weight.size()) == 4:
                            # matrix = torch.flatten(weight, start_dim=2) * math.sqrt(conv_norm)
                            # matrix = matrix.transpose(1, 2).transpose(0, 1)
                            matrix = weight.view(weight.shape[0], -1)
                        else:
                            matrix = weight

                        U, S, V = torch.svd(matrix)
                        U_, S_, V_ = U[:, :filter_rank], S[:filter_rank], V[:, :filter_rank]
                        SV = S_.unsqueeze(-1) * V_.T
                        W_ = torch.matmul(U_, SV)

                        W_[~prev_mask.bool()] = 0
                        flatten = torch.abs(W_).flatten()
                        if 'classifier' in name:
                            num_params_to_keep = int(len(flatten) * 0.2)
                        else:
                            num_params_to_keep = int(len(flatten) * (1 - curr_prune_rate))

                        top_k_indices = torch.topk(flatten, num_params_to_keep, sorted=True).indices
                        mask = torch.zeros_like(flatten, dtype=torch.bool)
                        mask[top_k_indices] = True
                        mask = mask.view(W_.shape).float()
                        if len(weight.size()) == 4:
                            O, I, H, W = weight.shape
                            self.masks[name] = mask.view(O, I, H, W).float()
                        else:
                            self.masks[name] = mask.view(weight.shape).float()

            else:
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        filter_rank = min(weight.shape[0] - 1, weight.shape[1] - 1, self.args.filter_rank)
                        prev_mask = self.masks[name].view(weight.shape[0], -1)
                        if len(weight.size()) == 4:
                            # matrix = torch.flatten(weight, start_dim=2) * math.sqrt(conv_norm)
                            # matrix = matrix.transpose(1, 2).transpose(0, 1)
                            matrix = weight.view(weight.shape[0], -1)
                        else:
                            matrix = weight

                        U, S, V = torch.svd(matrix)
                        U_, S_, V_ = U[:, :filter_rank], S[:filter_rank], V[:, :filter_rank]
                        SV = S_.unsqueeze(-1) * V_.T
                        W_ = torch.matmul(U_, SV)

                        W_[~prev_mask.bool()] = 0
                        flatten = torch.abs(W_).flatten()
                        num_params_to_keep = int(len(flatten) * (1 - curr_prune_rate))

                        top_k_indices = torch.topk(flatten, num_params_to_keep, sorted=True).indices
                        mask = torch.zeros_like(flatten, dtype=torch.bool)
                        mask[top_k_indices] = True
                        mask = mask.view(W_.shape).float()
                        if len(weight.size()) == 4:
                            O, I, H, W = weight.shape
                            self.masks[name] = mask.view(O, I, H, W).float()
                        else:
                            self.masks[name] = mask.view(weight.shape).float()

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format(
                (total_size-sparse_size) / total_size))

    def pruning_uniform(self, step):
        # prune_rate = 1 - self.args.final_density - self.args.init_density
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = (self.args.final_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps
        ini_iter = (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')

        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)
            # keep the density of the last layer as 0.2 if spasity is larger then 0.8
            if curr_prune_rate >= 0.8:
                curr_prune_rate = 1 - (self.total_params * (1-curr_prune_rate) - 0.2 * self.fc_params)/(self.total_params-self.fc_params)

                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        score = torch.flatten(torch.abs(weight))
                        if 'classifier' in name:
                            num_params_to_keep = int(len(score) * 0.2)
                            threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                            acceptable_score = threshold[-1]
                            self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                        else:
                            num_params_to_keep = int(len(score) * (1 - curr_prune_rate))
                            threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                            acceptable_score = threshold[-1]
                            self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
            else:
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        score = torch.flatten(torch.abs(weight))
                        num_params_to_keep = int(len(score) * (1 - curr_prune_rate))
                        threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                        acceptable_score = threshold[-1]
                        self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format(
                (total_size-sparse_size) / total_size))


    def add_module(self, module, sparse_init='ERK', grad_dic=None):
        self.module = module
        self.sparse_init = self.sparse_init
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                self.names.append(name)
                self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        if self.args.rm_first:
            for name, tensor in module.named_parameters():
                if 'conv.weight' in name or 'feature.0.weight' in name:
                    self.masks.pop(name)
                    print(f"pop out {name}")

        self.init(mode=self.args.sparse_init, density=self.args.init_density, grad_dict=grad_dic)


    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    for optim in self.optimizer:
                        if tensor in optim.state:
                            if 'momentum_buffer' in optim.state[tensor]:
                                optim.state[tensor]['momentum_buffer'] = optim.state[tensor]['momentum_buffer']*self.masks[name]


    def truncate_weights(self, step=None):

        self.gather_statistics()

        # prune
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]

                new_mask = self.magnitude_death(mask, weight, name)
                self.pruning_rate[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask

        # grow
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                new_mask = self.gradient_growth(name, new_mask, self.pruning_rate[name], weight)

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

        self.apply_mask()


    '''
                    REDISTRIBUTION
    '''

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]

                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                    #DEATH

    def magnitude_death(self, mask, weight, name):
        num_remove = math.ceil(self.prune_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)


                    #GROWTH

    def random_growth(self, name, new_mask, total_regrowth, weight):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability #lsw
        # new_weights = torch.rand(new_mask.shape) < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask


    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask



    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        for optim in self.optimizer:
            if weight in optim.state:
                if 'exp_avg' in optim.state[weight]:
                    adam_m1 = optim.state[weight]['exp_avg']
                    adam_m2 = optim.state[weight]['exp_avg_sq']
                    grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
                elif 'momentum_buffer' in optim.state[weight]:
                    grad = optim.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)

        print('Death rate: {0}\n'.format(self.prune_rate))

    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                for optim in self.optimizer:
                    if tensor in optim.state:
                        weights = list(optim.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        # momentum
                        if self.args.reset_mom_zero:
                            print('zero')
                            for optim in self.optimizer:
                                if tensor in optim.state:
                                    optim.state[tensor][w][mask == 0] = 0
                        else:
                            print('mean')
                            for optim in self.optimizer:
                                if tensor in optim.state:
                                    optim.state[tensor][w][mask == 0] = torch.mean(optim.state[tensor][w][mask.byte()])
                        # self.optimizer.state[tensor][w][mask==0] = 0
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        # Adam
                        for optim in self.optimizer:
                            if tensor in optim.state:
                                optim.state[tensor][w][mask==0] = torch.mean(optim.state[tensor][w][mask.byte()])

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights
