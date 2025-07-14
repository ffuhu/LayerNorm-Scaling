import os
import re
import glob
import matplotlib.pyplot as plt

# GATHER LOSSES OF EVERY MODEL WITHOUT PRUNING


logs_path = '../logs_server/'
files = glob.glob(logs_path + 'ew*/**/*.txt', recursive=True)

# remove pruning log files
files = sorted([f for f in files if 'pruned' not in f])

dict_losses = {}

for log_path in files:
    if not os.path.isfile(log_path): continue
    with open(log_path, 'r') as f:
        text = f.readlines()
        text = '\n'.join(text)

        optimizer = re.search(r'__([a-zA-Z_]+)_lr', text).group(1)
        lr = re.search(r'_lr([0-9.]+)', text).group(1)
        try:
            eval_loss = re.search(r'Final eval loss: ([0-9]{1,4}\.[0-9]+)', text).group(1)
        except Exception:
            eval_loss = 'ERROR'

        if optimizer not in dict_losses:
            dict_losses[optimizer] = {}

        dict_losses[optimizer][lr] = eval_loss


# Create the plot
plt.figure(figsize=(12, 8))

# Colors for each optimizer
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Plot each optimizer
for i, (optimizer, results) in enumerate(dict_losses.items()):
    learning_rates = []
    accuracies = []

    for lr, acc in results.items():
        if acc != 'ERROR':  # Skip ERROR values
            learning_rates.append(float(lr))
            accuracies.append(float(acc))

    # Sort by learning rate for proper line connection
    sorted_data = sorted(zip(learning_rates, accuracies))
    learning_rates, accuracies = zip(*sorted_data)

    plt.plot(learning_rates, accuracies,
             marker=markers[i],
             color=colors[i],
             label=optimizer,
             linewidth=2,
             markersize=8)

plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Accuracy vs Learning Rate by Optimizer', fontsize=14, fontweight='bold')
plt.xscale('log')  # Log scale for learning rates
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Add some styling
plt.tight_layout()
plt.show()


# GATHER LOSSES OF EVERY MODEL AFTER PRUNING


# with open('../logs_server/slurm.aisurrey25.909773.err') as f:
#     text = f.readlines()
#
# dict_losses_after_pruning = {}
# for line in text:
#     if "Saving model to" in line and "update steps" in line:
#
#         # Extract optimizer
#         pattern_optim = r'__([a-zA-Z_]+)_lr'
#         optimizer = re.search(pattern_optim, line).group(1)
#
#         # Extract learning rate
#         pattern_lr = r'_lr([0-9.]+)'
#         lr = re.search(pattern_lr, line).group(1)
#
#         # Extract the sparsity
#         pattern_sparsity = r'sparsity_([0-9.]+)'
#         sparsity = re.search(pattern_sparsity, line).group(1)
#
#         if optimizer not in dict_losses_after_pruning:
#             dict_losses_after_pruning[optimizer] = {}
#         if lr not in dict_losses_after_pruning[optimizer]:
#             dict_losses_after_pruning[optimizer][lr] = {}
#         if sparsity not in dict_losses_after_pruning[optimizer][lr]:
#             dict_losses_after_pruning[optimizer][lr][sparsity] = {}
#
#         pass
#
#     if "Eval loss before pruning" in line:
#         dict_losses_after_pruning[optimizer][lr][sparsity]['before'] = float(line.strip().split(' ')[-1])
#
#     if "Final eval loss" in line:
#         dict_losses_after_pruning[optimizer][lr][sparsity]['after'] = float(line.strip().split(' ')[-1])

logs_path = '../logs_server/'
files = glob.glob(logs_path + 'ew*/**/*.txt', recursive=True)

# remove pruning log files
files = sorted([f for f in files if 'log_pruning' in f])

dict_losses_after_pruning = {}

for log_path in files:
    if not os.path.isfile(log_path): continue
    with open(log_path, 'r') as f:
        text = f.readlines()
        text = '\n'.join(text)

        optimizer = re.search(r'__([a-zA-Z_]+)_lr', text).group(1)
        lr = re.search(r'_lr([0-9.]+)', text).group(1)
        sparsity = re.search(r'sparsity_([0-9.]+)', text).group(1)
        try:
            eval_loss_after = re.search(r'Final eval loss: ([0-9]{1,4}\.[0-9]+)', text).group(1)
        except Exception:
            eval_loss_after = 'ERROR'

        try:
            eval_loss_before = re.search(r'Eval loss before pruning: ([0-9]{1,4}\.[0-9]+)', text).group(1)
        except Exception:
            eval_loss_before = 'ERROR'

        if optimizer not in dict_losses_after_pruning:
            dict_losses_after_pruning[optimizer] = {}
        if lr not in dict_losses_after_pruning[optimizer]:
            dict_losses_after_pruning[optimizer][lr] = {}

        dict_losses_after_pruning[optimizer][lr]['before'] = eval_loss_before
        dict_losses_after_pruning[optimizer][lr][sparsity] = eval_loss_after

# print(dict_losses_after_pruning)

# Create plots for each optimizer
for optimizer_name, optimizer_data in dict_losses_after_pruning.items():
    learning_rates = list(optimizer_data.keys())
    n_lr = len(learning_rates)

    # Create subplots - 2 rows, 2 columns for 4 learning rates
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{optimizer_name.upper()} - Loss Before vs After Pruning', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes = axes.flatten()

    for i, lr in enumerate(learning_rates):
        ax = axes[i]

        # Extract data for this learning rate
        pruning_levels = []
        before_losses = []
        after_losses = []

        for pruning_level, losses in optimizer_data[lr].items():
            pruning_levels.append(float(pruning_level))
            before_losses.append(losses['before'])
            after_losses.append(losses['after'])

        # Sort by pruning level
        sorted_data = sorted(zip(pruning_levels, before_losses, after_losses))
        pruning_levels, before_losses, after_losses = zip(*sorted_data)

        # Plot before and after losses
        ax.plot(pruning_levels, before_losses, '--', label='Before Pruning',
                color='#2ca02c', linewidth=2), #, markersize=8)
        ax.plot(pruning_levels, after_losses, 's-', label='After Pruning',
                color='#d62728', linewidth=2, markersize=8)

        ax.set_title(f'Learning Rate: {lr}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Pruning Level', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set x-axis to show all pruning levels
        ax.set_xticks(pruning_levels)
        ax.set_xticklabels([f'{p:.1f}' for p in pruning_levels])

    plt.tight_layout()
    plt.show()