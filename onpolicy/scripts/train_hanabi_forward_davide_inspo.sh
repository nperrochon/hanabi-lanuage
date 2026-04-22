#!/bin/bash
#SBATCH --partition=ava_f.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --job-name=hanabi_train
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Load environment
source ~/.bashrc
conda activate myenv

# Debug info (helps a LOT when things go wrong)
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting job at: $(date)"

# Run training
python onpolicy/scripts/train/train_hanabi_forward.py \
  --cuda \
  --n_rollout_threads 8 \
  --n_training_threads 1

echo "Finished at: $(date)"
