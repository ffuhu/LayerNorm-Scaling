#!/bin/bash

#SBATCH -p gpu_a100
#SBATCH -N 1
##SBATCH --array=3-6 <-- SET IT WHEN CALLING "sbtach --array=N-M file.sh"
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
##SBATCH -t 01:15:00
##SBATCH -t 03:15:00  ## for SOAP
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/0/prjs1462/ffuhu/repos/PrunOptNew/logs/PrunOpt/slurm_cifar_full_resnet20_slurm.%N.%j.out
#SBATCH --error=/projects/0/prjs1462/ffuhu/repos/PrunOptNew/logs/PrunOpt/slurm_cifar_full_resnet20_slurm.%N.%j.err


export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME=/projects/0/prjs1462/ffuhu/cache/huggingface

pwd
hostname
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo starting job...

SRC_DIR=/projects/0/prjs1462/ffuhu/repos/PrunOptNew/
CKPT_DIR=/projects/0/prjs1462/ffuhu/repos/PrunOptNew/checkpoints/full_wwd_wowd/

#SLURM_ARRAY_TASK_ID=$1
echo $SLURM_ARRAY_TASK_ID
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/base_configs_resnet20.txt)
echo $cfg
optimizer=$(echo $cfg | cut -f 1 -d ' ')
l2=$(echo $cfg | cut -f 2 -d ' ')
lr=$(echo $cfg | cut -f 3 -d ' ')
lr_scheduler=$(echo $cfg | cut -f 4 -d ' ')
#lr_min=$(echo $cfg | cut -f 5 -d ' ')
epochs=$(echo $cfg | cut -f 5 -d ' ')
model=$(echo $cfg | cut -f 6 -d ' ')
data=$(echo $cfg | cut -f 7 -d ' ')
seed=$(echo $cfg | cut -f 8 -d ' ')
eval_hessian=$(echo $cfg | cut -f 9 -d ' ')
precondition_frequency=$(echo $cfg | cut -f 10 -d ' ')
muon_lr=$(echo $cfg | cut -f 11 -d ' ')

if [[ "${optimizer:0:1}" == "#" ]]; then
  exit 0  # training already done
fi

if [ "$precondition_frequency" = "" ]; then
  precondition_frequency=-1
fi

if [ "$muon_lr" = "" ]; then
  muon_lr_arg=""
else
  muon_lr_arg="--muon_lr ${muon_lr}"
fi

cd $SRC_DIR

source /projects/0/prjs1462/miniforge3/bin/activate
conda activate spam

#conda run -n spam huggingface-cli login --token $HF_TOKEN
#echo $HF_HOME

conda run -n spam huggingface-cli whoami

echo "Running the script:
conda run -n spam python main.py \
    --optimizer ${optimizer} \
    --l2 ${l2} \
    --lr ${lr} \
    --lr_scheduler ${lr_scheduler} \
    --epochs ${epochs} \
    --model ${model} \
    --data ${data} \
    --save_path ${CKPT_DIR} \
    --seed ${seed} \
    --eval_hessian ${eval_hessian} \
    --precondition_frequency ${precondition_frequency} \
    ${muon_lr_arg}"

conda run -n spam python main.py \
    --optimizer ${optimizer} \
    --l2 ${l2} \
    --lr ${lr} \
    --lr_scheduler ${lr_scheduler} \
    --epochs ${epochs} \
    --model ${model} \
    --data ${data} \
    --save_path ${CKPT_DIR} \
    --seed ${seed} \
    --eval_hessian ${eval_hessian} \
    --precondition_frequency ${precondition_frequency} \
    ${muon_lr_arg}

#conda run -n spam huggingface-cli logout
