#!/bin/bash
#SBATCH --partition=ava_f.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=hanabi_qwen3_perep
#SBATCH --output=logs/slurm_qwen3_perep_%j.out   # Swapped to standard %j Job ID token
#SBATCH --error=logs/slurm_qwen3_perep_%j.err

# =========================
# Environment setup
# =========================
set +u
source ~/.bashrc
conda activate qwen3_embed
set -euo pipefail

REPO="${HOME}/hanabi-lanuage"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

cd "${REPO}"

PYTHON="/home/nperroch/.conda/envs/qwen3_embed/bin/python"
export PATH="/home/nperroch/.conda/envs/qwen3_embed/bin:$PATH"

unset HF_ENDPOINT
export HF_HOME="${HOME}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=online
export WANDB_SILENT=false
export WANDB_CONSOLE=off
export WANDB_DIR="${REPO}/wandb"
mkdir -p "${WANDB_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Started at: $(date)"

nvidia-smi || true

# =========================
# Fixed Hanabi / PPO settings
# =========================
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=8
seed=1
num_mini_batch=1

# =========================
# Hardcoded Qwen3 Target Configuration
# =========================
exp="qwen3_embed_8threads_p010_perep_seed1" # Added _perep for clean WandB tracking
use_llm="1"
llm_model="Qwen/Qwen3-Embedding-0.6B"
llm_backend="qwen_embedding"
llm_vector_mode="qwen_embedding"
llm_step_prob="0.10"

echo "Experiment: ${exp}"
echo "Use LLM: ${use_llm}"
echo "LLM model: ${llm_model}"
echo "LLM backend: ${llm_backend}"
echo "LLM vector mode: ${llm_vector_mode}"
echo "LLM step prob: ${llm_step_prob}"

# Secure port mapping scaled against standard Job ID to prevent socket overlaps
EMBED_HOST="127.0.0.1"
EMBED_PORT=$((25000 + SLURM_JOB_ID % 1000))

# =========================
# Embedding server
# =========================
echo "Starting embedding server on ${EMBED_HOST}:${EMBED_PORT}..."

"${PYTHON}" -u onpolicy/envs/hanabi/qwen_embedding_server.py \
  --model "${llm_model}" \
  --host "${EMBED_HOST}" \
  --port "${EMBED_PORT}" \
  --device cuda \
  > "${LOG_DIR}/embed_server_${exp}_${SLURM_JOB_ID}.log" 2>&1 &

EMBED_PID=$!

cleanup() {
  echo "Cleaning up embedding server PID=${EMBED_PID}"
  kill "${EMBED_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for embedding server..."
for i in {1..120}; do
  # FIXED: Included the structural whitespace gap inside python's health output string match
  if curl -s "http://${EMBED_HOST}:${EMBED_PORT}/health" | tr -d ' ' | grep -iq '"ok":true'; then
    echo "Embedding server ready after ${i} checks"
    curl -s "http://${EMBED_HOST}:${EMBED_PORT}/health"
    echo ""
    break
  fi

  if ! kill -0 "${EMBED_PID}" 2>/dev/null; then
    echo "Embedding server died. Last log lines:"
    tail -100 "${LOG_DIR}/embed_server_${exp}_${SLURM_JOB_ID}.log" || true
    exit 1
  fi

  echo "Still waiting for embedding server... check ${i}"
  sleep 5
done

# =========================
# Build training command
# =========================
CMD=(
  "${PYTHON}" -u onpolicy/scripts/train/train_hanabi_forward.py
  --env_name "${env}"
  --algorithm_name "${algo}"
  --experiment_name "${exp}"
  --hanabi_name "${hanabi}"
  --num_agents "${num_agents}"
  --seed "${seed}"
  --n_training_threads 1
  --n_rollout_threads "${rollout_threads}"
  --num_mini_batch "${num_mini_batch}"
  --episode_length 100
  --num_env_steps 1000000000
  --ppo_epoch 10
  --gain 0.01
  --lr 7e-4
  --critic_lr 1e-3
  --clip_param 0.1
  --hidden_size 512
  --layer_N 2
  --entropy_coef 0.015
  --log_interval 5
  --save_every_x_updates 200
  --use_llm
  --llm_model "${llm_model}"
  --llm_backend "${llm_backend}"
  --llm_vector_mode "${llm_vector_mode}"
  --llm_step_prob "${llm_step_prob}"
  --qwen_embed_host "${EMBED_HOST}"
  --qwen_embed_port "${EMBED_PORT}"
)

echo "Starting Python training at: $(date)"
echo "Command:"
printf '%q ' "${CMD[@]}"
echo ""

"${CMD[@]}"

echo "Job ended at: $(date)"
