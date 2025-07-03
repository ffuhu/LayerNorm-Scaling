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
#norm_type="LNS" #$1
#learning_rate=1e-3
export NORM_TYPE="LNS" #$norm_type
#export POST_NUM=3 #$2



#TODO: make if else to run sgd, adam and adam mini depending on $1 for slurm array

case "$1" in
  1)
    echo "You chose ONE."
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=1e-4
    ;;
  2)
    echo "You chose TWO."
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=5e-4
    ;;
  3)
    echo "You chose THREE."
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=1e-3
    ;;
  4)
    echo "You chose FOUR."
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    ;;
  *)
    echo "Number not recognized. Please enter 1, 2, or 3."
    exit
    ;;
esac


# Function to run a single training task
echo "Training with $optimizer (learning rate: $learning_rate, weight decay: $weight_weight_decay) norm type: $NORM_TYPE on GPU $gpu"

#conda run torchrun --nproc_per_node 1 --master_port=29510 torchrun_main.py \
#    --model_config configs/llama_130m.json \
#    --lr $learning_rate \
#    --batch_size 32 \
#    --total_batch_size 64 \
#    --num_training_steps 160000 \
#    --warmup_steps 2000 \
#    --weight_decay 0 \
#    --dtype bfloat16 \
#    --eval_every 1000 \
#    --save_every 1000 \
#    --optimizer $optimizer \
#    --weight_decay $weight_decay \
#    --grad_clipping 0.0 \
#    --run_name "ew_130m_save0-5-11_${norm_type}_lr${learning_rate}" \
#    --save_dir "logs" \
#    --layers_to_save layers.0 layers.5 layers.11 \
#    --save_every_N_steps 10 \
#    --beta1 0.98 \
#    --weight_decay $weight_decay

echo "torchrun --nproc_per_node 1 --master_port=29510 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $learning_rate \
    --batch_size 32 \
    --total_batch_size 64 \
    --num_training_steps 160000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 1000 \
    --optimizer $optimizer \
    --weight_decay $weight_decay \
    --grad_clipping 0.0 \
    --run_name "ew_130m_save0-5-11_${norm_type}_lr${learning_rate}" \
    --save_dir "logs" \
    --layers_to_save layers.0 layers.5 layers.11 \
    --save_every_N_steps 10 \
    --beta1 0.98 \
    --weight_decay $weight_decay"






