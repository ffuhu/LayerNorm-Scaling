#!/bin/bash

#SBATCH -p gpu_a100
#SBATCH -N 1
##SBATCH --array=3-6 <-- SET IT WHEN CALLING "sbtach --array=N-M file.sh"
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 02:00:00
##SBATCH -t 03:15:00  ## for SOAP
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/0/prjs1462/ffuhu/repos/LayerNorm-Scaling/logs/slurm.%N.%j.out
#SBATCH --error=/projects/0/prjs1462/ffuhu/repos/LayerNorm-Scaling/logs/slurm.%N.%j.err


export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

source /projects/0/prjs1462/miniforge3/bin/activate
conda activate spam

# Define the set of learning rates and normalization types
norm_type="LNS" #$1
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=3 #$2

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

conda run torchrun --standalone --nproc_per_node 1 --master_port=29510 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $learning_rates \
    --batch_size 32 \
    --total_batch_size 64 \
    --num_training_steps 160000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "ew_130m_res_${norm_type}_lr${learning_rates}_layer_scale" \
    --save_dir "ew_130m_res_${norm_type}_lr${learning_rates}"






