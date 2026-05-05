#!/bin/bash
#SBATCH --job-name=hanabi_mappo
#SBATCH --output=/data/class/mae93/nperroch/hanabi-lanuage/logs/hanabi_%j.out
#SBATCH --error=/data/class/mae93/nperroch/hanabi-lanuage/logs/hanabi_%j.err
#SBATCH --partition=free-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=8
llm_model="qwen2:7b"
exp="single_run_${rollout_threads}threads_${llm_model}"
seed=1

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
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "start time: $(date)"

cd ~/hanabi-lanuage || exit 1

ollama serve > /home/nperroch/hanabi-lanuage/logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
sleep 10

ollama pull qwen2:7b
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
ollama pull phi3:mini
ollama pull gemma3:4b

sleep 10
echo "warming up model..."
ollama run ${llm_model} "Say 1" > /dev/null
ollama list

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
  --ppo_epoch 10 \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 1e-3 \
  --clip_param 0.1 \
  --hidden_size 512 \
  --layer_N 2 \
  --entropy_coef 0.015 \
  --log_interval 5 \
  --use_llm \
  --llm_model ${llm_model} \
  --llm_step_prob 0.02 \


echo "end time: $(date)"

