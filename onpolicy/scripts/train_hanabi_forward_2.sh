#!/bin/bash
#SBATCH --job-name=hanabi_mappo
#SBATCH --output=/home/nperroch/hanabi-lanuage/logs/hanabi_%j.out
#SBATCH --error=/home/nperroch/hanabi-lanuage/logs/hanabi_%j.err
#SBATCH --partition=ava_f.p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=8
exp="single_run_${rollout_threads}threads"
seed=1

mkdir -p /home/nperroch/hanabi-lanuage/logs
ulimit -n 22222

echo "env=${env}"
echo "algo=${algo}"
echo "exp=${exp}"
echo "seed=${seed}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "start time: $(date)"

cd ~/hanabi-lanuage || exit 1

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
  --num_env_steps 1000000000 \
  --ppo_epoch 6 \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 1e-3 \
  --clip_param 0.1 \
  --hidden_size 512 \
  --layer_N 2 \
  --entropy_coef 0.01 \
  --log_interval 5

echo "end time: $(date)"
