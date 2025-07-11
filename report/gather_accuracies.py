import os
import re

with open('/media/felix/d519d0a7-de67-4175-989e-1730f5d95c14/Scratch/shiwei/LayerNorm-Scaling/logs_server/slurm.aisurrey25.909773.err') as f:
    text = f.readlines()

accuracies = {}
for line in text:
    if "Saving model to" in line and "update steps" in line:

        # Extract optimizer
        pattern_optim = r'__([a-zA-Z_]+)_lr'
        optimizer = re.search(pattern_optim, line).group(1)

        # Extract learning rate
        pattern_lr = r'_lr([0-9.]+)'
        lr = re.search(pattern_lr, line).group(1)

        # Extract the sparsity
        pattern_sparsity = r'sparsity_([0-9.]+)'
        sparsity = re.search(pattern_sparsity, line).group(1)

        if optimizer not in accuracies:
            accuracies[optimizer] = {}
        if lr not in accuracies[optimizer]:
            accuracies[optimizer][lr] = {}
        if sparsity not in accuracies[optimizer][lr]:
            accuracies[optimizer][lr][sparsity] = {}

        pass

    if "Eval loss before pruning" in line:
        accuracies[optimizer][lr][sparsity]['before'] = float(line.strip().split(' ')[-1])

    if "Final eval loss" in line:
        accuracies[optimizer][lr][sparsity]['after'] = float(line.strip().split(' ')[-1])

pass





