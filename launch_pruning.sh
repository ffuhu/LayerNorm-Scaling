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
export NORM_TYPE="LNS" #$norm_type

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

export MASTER_ADDR='localhost'
master_port=$((29500+SLURM_ARRAY_TASK_ID))
export MASTER_PORT=$master_port
echo "Master port: $MASTER_PORT"

#cd /mnt/fast/nobackup/scratch4weeks/ly0008/ffuhu/LayerNorm-Scaling/
echo "Working directory:"
pwd

# Function to run a single training task
echo "PRUNING...."

# sparsities
sparsities=(0.05 0.1 0.2 0.3 0.4 0.5)

# checkpoints
ckpts=(
#./logs/ew_130m_save0-5-11__adam_mini_lr0.0001_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adam_mini_lr0.0005_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adam_mini_lr0.001_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adam_mini_lr0.003_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adamw_lr0.0001_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adamw_lr0.0005_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adamw_lr0.001_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adamw_lr0.003_wd0.1_seed1/model_160001/
./logs/ew_130m_save0-5-11__muon_lr0.0001_wd0.0005_seed1/model_160001/
./logs/ew_130m_save0-5-11__muon_lr0.0005_wd0.0005_seed1/model_160001/
./logs/ew_130m_save0-5-11__muon_lr0.001_wd0.0005_seed1/model_160001/
./logs/ew_130m_save0-5-11__muon_lr0.003_wd0.0005_seed1/model_160001/
./logs/ew_130m_save0-5-11__sgd_lr0.01_wd0.0005_seed1/model_160001/
./logs/ew_130m_save0-5-11__sgd_lr0.05_wd0.0005_seed1/model_160001/
./logs/ew_130m_save0-5-11__sgd_lr0.1_wd0.0005_seed1/model_160001/
./logs/ew_130m_save0-5-11__sgd_lr0.5_wd0.0005_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adam_mini_lr0.1_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adam_mini_lr0.01_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adamw_lr0.1_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__adamw_lr0.01_wd0.1_seed1/model_160001/
#./logs/ew_130m_save0-5-11__muon_lr0.1_wd0.0005_seed1/model_160001/
#./logs/ew_130m_save0-5-11__muon_lr0.01_wd0.0005_seed1/model_160001/

)

for ckpt in "${ckpts[@]}"; do
  for sparsity in "${sparsities[@]}"; do
    echo "Running $ckpt with sparsity $sparsity"
    conda run -n cod python torchrun_prune.py \
        --model_config configs/llama_130m.json \
        --batch_size 32 \
        --dtype bfloat16 \
        --save_dir $ckpt \
        --continue_from $ckpt  \
        --sparsity $sparsity
#    python torchrun_prune.py \
#        --model_config configs/llama_130m.json \
#        --batch_size 32 \
#        --dtype bfloat16 \
#        --save_dir $ckpt \
#        --continue_from $ckpt  \
#        --sparsity $sparsity
    echo "conda run -n cod python torchrun_prune.py \
        --model_config configs/llama_130m.json \
        --batch_size 32 \
        --dtype bfloat16 \
        --save_dir $ckpt \
        --continue_from $ckpt  \
        --sparsity $sparsity"
  done
done