#!/bin/sh
#SBATCH --job-name=replicate_mappo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G              ## Increased memory for stability
#SBATCH --time=48:00:00        ## Hanabi-Full takes significant time
#SBATCH --partition=standard
#SBATCH --account=mae93_class ## Found via 'sshare -U $USER'

ulimit -n 22222
# Paper-aligned Hanabi settings (Yu et al., 2022)
rollout_threads=128            ## Scaled down from 1000 for 16 requested CPUs
training_threads=16

# Add this before the python call
for seed in 1; do
    python train/train_hanabi_forward.py \
        --env_name "Hanabi" --algorithm_name "mappo" --experiment_name "replication" \
        --hanabi_name "Hanabi-Full" --num_agents 2 --seed ${seed} \
        --n_training_threads ${training_threads} \
        --n_rollout_threads ${rollout_threads} \
        --num_mini_batch 1 \
        --episode_length 100 \
        --ppo_epoch 15 \
        --gain 0.01 --lr 7e-4 --critic_lr 1e-3 \
        --hidden_size 512 --layer_N 2 \
        --entropy_coef 0.015 \
        --use_valuenorm False --use_popart True \
        --num_env_steps 1000000000 \
        --device cpu 2>&1 | tee -a train_log_seed_${seed}.txt
done
