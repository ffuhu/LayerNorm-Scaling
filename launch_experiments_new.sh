#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
##SBATCH --array=3-6 <-- SET IT WHEN CALLING "sbtach --array=N-M file.sh"
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 02:00:00
##SBATCH -t 03:15:00  ## for SOAP
#SBATCH --cpus-per-task=16
##SBATCH --output=/projects/0/prjs1462/ffuhu/repos/LayerNorm-Scaling/logs/slurm.%N.%j.out
##SBATCH --error=/projects/0/prjs1462/ffuhu/repos/LayerNorm-Scaling/logs/slurm.%N.%j.err
#SBATCH --output=./logs/slurm.%N.%j.out
#SBATCH --error=./logs/slurm.%N.%j.err


export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
#export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
#export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

#source /projects/0/prjs1462/miniforge3/bin/activate
#conda activate ew

# Define the set of learning rates and normalization types
#norm_type="LNS" #$1
#learning_rate=1e-3
export NORM_TYPE="LNS" #$norm_type
#export POST_NUM=3 #$2

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

export MASTER_ADDR='localhost'
master_port=$((29500+SLURM_ARRAY_TASK_ID))
export MASTER_PORT=$master_port
echo "Master port: $MASTER_PORT"

num_training_steps=160000
warmup_steps=2000
#batch_size=32
#total_batch_size=64
beta1=0.95
beta2=0.95
momentum=0.98
case "$SLURM_ARRAY_TASK_ID" in
  # grid search sgd and adam
  50)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=64
    total_batch_size=64
    # num_training_steps=40000
    # warmup_steps=2000
    ;;
  51)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=32
    total_batch_size=32
    # num_training_steps=80000
    # warmup_steps=2000
    ;;
  52)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=16
    total_batch_size=16
    # num_training_steps=160000
    # warmup_steps=2000
    ;;
  53)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=8
    total_batch_size=8
    # num_training_steps=320000
    # warmup_steps=2000
    ;;
  54)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=64
    total_batch_size=64
    # num_training_steps=40000
    # warmup_steps=2000
    ;;
  55)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=32
    total_batch_size=32
    # num_training_steps=80000
    # warmup_steps=2000
    ;;
  56)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=16
    total_batch_size=16
    # num_training_steps=160000
    # warmup_steps=2000
    ;;
  57)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=8
    total_batch_size=8
    # num_training_steps=320000
    # warmup_steps=2000
    ;;
  60)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=512
    total_batch_size=1024
    # num_training_steps=40000
    # warmup_steps=2000
    ;;
  61)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=512
    total_batch_size=512
    # num_training_steps=80000
    # warmup_steps=2000
    ;;
  62)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=256
    total_batch_size=256
    # num_training_steps=160000
    # warmup_steps=2000
    ;;
  63)
    optimizer="sgd"
    weight_decay=0.0005
    learning_rate=1e-1
    batch_size=128
    total_batch_size=128
    # num_training_steps=320000
    # warmup_steps=2000
    ;;
  64)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=512
    total_batch_size=1024
    # num_training_steps=2500
    # warmup_steps=2000
    ;;
  65)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=512
    total_batch_size=512
    # num_training_steps=80000
    # warmup_steps=2000
    ;;
  66)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=256
    total_batch_size=256
    # num_training_steps=160000
    # warmup_steps=2000
    ;;
  67)
    optimizer="adamw"
    weight_decay=0.1
    learning_rate=3e-3
    batch_size=128
    total_batch_size=128
    # num_training_steps=320000
    # warmup_steps=2000
    ;;
  *)
    echo "Number not recognized. Please enter 1-16."
    exit
    ;;
esac

#cd /mnt/fast/nobackup/scratch4weeks/ly0008/ffuhu/LayerNorm-Scaling/
echo "Working directory:"
pwd

# Function to run a single training task
echo "Training with $optimizer (learning rate: $learning_rate, weight decay: $weight_weight_decay) norm type: $NORM_TYPE on GPU $gpu"

conda run -n cod torchrun --nproc_per_node 1 --master_port=$master_port torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $learning_rate \
    --batch_size $batch_size \
    --total_batch_size $total_batch_size \
    --num_training_steps $num_training_steps \
    --warmup_steps $warmup_steps \
    --dtype bfloat16 \
    --eval_every 10000 \
    --save_every 10000 \
    --optimizer $optimizer \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --momentum $momentum \
    --weight_decay $weight_decay \
    --grad_clipping 0.0 \
    --run_name "gs_ew_130m_save0-5-11_${norm_type}" \
    --save_dir "logs" \
    --layers_to_save layers.0 layers.5 layers.11 \
    --save_every_N_steps 10

echo "conda run -n cod torchrun --nproc_per_node 1 --master_port=$master_port torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $learning_rate \
    --batch_size $batch_size \
    --total_batch_size $total_batch_size \
    --num_training_steps $num_training_steps \
    --warmup_steps $warmup_steps \
    --dtype bfloat16 \
    --eval_every 10000 \
    --save_every 10000 \
    --optimizer $optimizer \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --momentum $momentum \
    --weight_decay $weight_decay \
    --grad_clipping 0.0 \
    --run_name "gs_ew_130m_save0-5-11_${norm_type}" \
    --save_dir "logs" \
    --layers_to_save layers.0 layers.5 layers.11 \
    --save_every_N_steps 10"
