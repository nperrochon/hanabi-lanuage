#!/bin/bash
#SBATCH --job-name=hanabi_qwen_embed
#SBATCH --output=/data/class/mae93/nperroch/hanabi-lanuage/logs/hanabi_qwen_embed_%j.out
#SBATCH --error=/data/class/mae93/nperroch/hanabi-lanuage/logs/hanabi_qwen_embed_%j.err
#SBATCH -A royf_lab_gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:A100:1

set -euo pipefail

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=8
seed=1

llm_model="Qwen/Qwen3-Embedding-0.6B"
llm_backend="qwen_embedding"
llm_vector_mode="qwen_embedding"
llm_step_prob=1.0

REPO="/data/class/mae93/nperroch/hanabi-lanuage"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

# Avoid port conflicts between jobs.
EMBED_PORT=$((8700 + SLURM_JOB_ID % 1000))
EMBED_HOST="127.0.0.1"

exp="qwen_embed_server_${rollout_threads}threads_step${llm_step_prob}_seed${seed}"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Experiment: ${exp}"
echo "Embedding server: ${EMBED_HOST}:${EMBED_PORT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marl

cd "${REPO}"

unset HF_ENDPOINT
export HF_HOME=/data/class/mae93/nperroch/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

nvidia-smi || true

echo "Starting Qwen embedding server..."
python -u onpolicy/envs/hanabi/qwen_embedding_server.py \
  --model "${llm_model}" \
  --host "${EMBED_HOST}" \
  --port "${EMBED_PORT}" \
  --device cuda \
  > "${LOG_DIR}/qwen_embed_server_${SLURM_JOB_ID}.log" 2>&1 &

EMBED_PID=$!

cleanup() {
  echo "Cleaning up embedding server PID=${EMBED_PID}"
  kill "${EMBED_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Qwen embedding server..."
for i in {1..120}; do
  if curl -s "http://${EMBED_HOST}:${EMBED_PORT}/health" | grep -iq '"ok": true'; then
    echo "Qwen embedding server ready after ${i} checks"
    curl -s "http://${EMBED_HOST}:${EMBED_PORT}/health"
    echo ""
    break
  fi

  if ! kill -0 "${EMBED_PID}" 2>/dev/null; then
    echo "Qwen embedding server died. Last log lines:"
    tail -100 "${LOG_DIR}/qwen_embed_server_${SLURM_JOB_ID}.log"
    exit 1
  fi

  echo "Still waiting for Qwen embedding server... check ${i}"
  sleep 5
done

echo "Starting Python training at: $(date)"

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
  --llm_backend ${llm_backend} \
  --llm_vector_mode ${llm_vector_mode} \
  --llm_step_prob ${llm_step_prob} \
  --qwen_embed_host ${EMBED_HOST} \
  --qwen_embed_port ${EMBED_PORT}

echo "Job ended at: $(date)"
