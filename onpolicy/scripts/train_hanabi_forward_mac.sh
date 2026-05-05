#!/bin/bash
#SBATCH --job-name=hanabi_mappo
#SBATCH --output=/home/nperroch/hanabi-lanuage/logs/hanabi_%j.out
#SBATCH --error=/home/nperroch/hanabi-lanuage/logs/hanabi_%j.err
#SBATCH --partition=ava_f.p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=4
exp="single_run_${rollout_threads}threads"
seed=1
deterministic_deal=false
model_dir="/home/nperroch/hanabi-lanuage/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo/single_run_8threads/wandb/run-20260420_202622-o2kis2qu/files"
model_dir_epoch_10="/home/nperroch/hanabi-lanuage/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo/single_run_8threads/wandb/run-20260420_130429-miejs3il/files"
eval_model_dir="/home/nperroch/hanabi-lanuage/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo/single_run_8threads/wandb/run-20260421_105027-dqry5qi9/files"

model_dir_continued="/home/nperroch/hanabi-lanuage/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo/single_run_8threads/wandb/run-20260421_105027-dqry5qi9/files"
model_dir_ppo_10_ent_0015="/home/nperroch/hanabi-lanuage/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo/single_run_8threads/wandb/run-20260420_125923-66sgqo0l/files"

mkdir -p /home/nperroch/hanabi-lanuage/logs
ulimit -n 22222

echo "env=${env}"
echo "algo=${algo}"
echo "exp=${exp}"
echo "seed=${seed}"
echo "deterministic_deal=${deterministic_deal}"
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
  $( [ "${deterministic_deal}" = "true" ] && echo "--deterministic_deal" ) \
  --n_training_threads 1 \
  --n_rollout_threads ${rollout_threads} \
  --num_mini_batch 1 \
  --episode_length 100 \
  --num_env_steps 1000000000 \
  --ppo_epoch 10 \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 1e-3 \
  --clip_param 0.1 \
  --hidden_size 512 \
  --layer_N 2 \
  --entropy_coef 0.015 \
  --log_interval 5 \

echo "end time: $(date)"

