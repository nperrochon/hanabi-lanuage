#!/bin/sh
#SBATCH --job-name=hanabi_mappo_eval
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
exp="single_run_8threads"
model_dir="/data/class/mae93/nperroch/hanabi-lanuage/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo/single_run_8threads_Qwen/Qwen2.5-1.5B-Instruct/wandb/run-20260508_014953-mscxdyj8/files"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python onpolicy/scripts/eval/eval_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --hanabi_name ${hanabi} --num_agents ${num_agents} --seed 1 --n_training_threads 128 --n_rollout_threads 1 \
    --n_eval_rollout_threads 10 --num_mini_batch 4 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 \
    --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --use_eval --use_llm --llm_backend vllm --llm_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --llm_base_url "http://127.0.0.1:8000/v1" \
    --llm_step_prob 0.02 --use_wandb --use_recurrent_policy \
    --entropy_coef 0.015 --model_dir ${model_dir}
    echo "training is done!"
done
