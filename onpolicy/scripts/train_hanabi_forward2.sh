#!/bin/bash
#SBATCH --job-name=hanabi_mappo
#SBATCH --output=logs/hanabi_%j.out
#SBATCH --error=logs/hanabi_%j.err
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=8
exp="hpc3_single_run_${rollout_threads}threads"
seed=1


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate marl
python - <<'PY'
import sys, torch
print("sys.executable =", sys.executable)
print("torch.__version__ =", torch.__version__)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
PY

mkdir -p logs
ulimit -n 22222

echo "env=${env}"
echo "algo=${algo}"
echo "exp=${exp}"
echo "seed=${seed}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "start time: $(date)"

cd ~/hanabi-lanuage || exit 1
conda activate marl

python -u onpolicy/scripts/train/train_hanabi_forward.py \
  --env_name ${env} \
  --algorithm_name ${algo} \
  --experiment_name ${exp} \
  --hanabi_name ${hanabi} \
  --num_agents ${num_agents} \
  --seed ${seed} \
  --n_training_threads 1 \
  --n_rollout_threads ${rollout_threads} \
  --num_mini_batch 1 \
  --episode_length 100 \
  --num_env_steps 2000000 \
  --ppo_epoch 15 \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 1e-3 \
  --hidden_size 512 \
  --layer_N 2 \
  --entropy_coef 0.015 \
  --log_interval 1

echo "end time: $(date)"
