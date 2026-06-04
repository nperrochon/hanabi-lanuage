#!/bin/bash
#SBATCH --partition=ava_f.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%6
#SBATCH --job-name=hanabi_eval_sweep
#SBATCH --output=logs/slurm_eval_cmp_%A_%a.out
#SBATCH --error=logs/slurm_eval_cmp_%A_%a.err

set -euo pipefail

REPO="${HOME}/hanabi-lanuage"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

cd "${REPO}"

# Environment Initialization
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3_embed

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
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Started at: $(date)"

nvidia-smi || true

# Fixed Hanabi / PPO settings
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=1
num_mini_batch=1

# Define configs
CONFIGS=(
  "baseline_no_llm_8threads_seed1|0|NONE|NONE|NONE|0.00"
  "qwen3_embed_8threads_p010_seed1|1|Qwen/Qwen3-Embedding-0.6B|qwen_embedding|qwen_embedding|0.10"
  "minilm_embed_8threads_p020_seed1|1|sentence-transformers/all-MiniLM-L6-v2|qwen_embedding|qwen_embedding|0.20"
  "bge_small_embed_8threads_p020_seed1|1|BAAI/bge-small-en-v1.5|qwen_embedding|qwen_embedding|0.20"
  "mpnet_embed_8threads_p020_seed1|1|sentence-transformers/all-mpnet-base-v2|qwen_embedding|qwen_embedding|0.20"
  "bge_large_embed_8threads_p020_seed1|1|BAAI/bge-large-en-v1.5|qwen_embedding|qwen_embedding|0.20"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r exp use_llm llm_model llm_backend llm_vector_mode train_prob <<< "${CONFIG}"

BASE_RESULTS_DIR="${REPO}/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo"

# ==============================================================================
# 🧠 AUTOMATED CHECKPOINT RESOLVER & SYMLINK INJECTOR
# Finds the highest version milestone file and creates standard named symlinks
# to fulfill the expectations of base_runner.py without altering saved logs.
# ==============================================================================
echo "Searching for active checkpoint files under: ${BASE_RESULTS_DIR}/${exp}"

set +e
# Collect paths and sort them version-wise to guarantee grabbing the latest milestone
ACTOR_FILES=($(find "${BASE_RESULTS_DIR}/${exp}" -name "actor*.pt" 2>/dev/null | sort -V))
set -e

if [ ${#ACTOR_FILES[@]} -gt 0 ]; then
    # Target the last element (the latest training milestone saved)
    LATEST_ACTOR="${ACTOR_FILES[-1]}"
    MODEL_DIR=$(dirname "${LATEST_ACTOR}")
    ACTOR_FILENAME=$(basename "${LATEST_ACTOR}")

    echo "SUCCESS: Discovered active checkpoint folder: ${MODEL_DIR}"
    echo "Targeting checkpoint state file: ${ACTOR_FILENAME}"

    # Generate explicit symlink for actor.pt
    ln -sf "${ACTOR_FILENAME}" "${MODEL_DIR}/actor.pt"

    # Calculate and link corresponding critic baseline profile
    CRITIC_FILENAME=$(echo "${ACTOR_FILENAME}" | sed 's/actor/critic/')
    if [ -f "${MODEL_DIR}/${CRITIC_FILENAME}" ]; then
        ln -sf "${CRITIC_FILENAME}" "${MODEL_DIR}/critic.pt"
    fi
else
    echo "WARNING: No actor*.pt files found. Defaulting to base experiment directory."
    MODEL_DIR="${BASE_RESULTS_DIR}/${exp}"
fi
# ==============================================================================

eval_exp="${exp}_EVAL_100PCT"
EMBED_HOST="127.0.0.1"
EMBED_PORT=$((24000 + (SLURM_ARRAY_JOB_ID % 500) * 10 + SLURM_ARRAY_TASK_ID))

# Optional embedding server
if [ "${use_llm}" = "1" ]; then
  echo "Starting background embedding server on ${EMBED_HOST}:${EMBED_PORT}..."

  "${PYTHON}" -u onpolicy/envs/hanabi/qwen_embedding_server.py \
    --model "${llm_model}" \
    --host "${EMBED_HOST}" \
    --port "${EMBED_PORT}" \
    --device cuda \
    > "${LOG_DIR}/eval_server_${eval_exp}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &

  EMBED_PID=$!

  cleanup() {
    echo "Cleaning up embedding server PID=${EMBED_PID}"
    kill "${EMBED_PID}" 2>/dev/null || true
  }
  trap cleanup EXIT

  echo "Waiting for embedding server validation..."
  for i in {1..120}; do
    if curl -s "http://${EMBED_HOST}:${EMBED_PORT}/health" | grep -iq '"ok": true'; then
      echo "Embedding server ready after ${i} checks"
      break
    fi

    if ! kill -0 "${EMBED_PID}" 2>/dev/null; then
      echo "Embedding server died. Last log lines:"
      tail -50 "${LOG_DIR}/eval_server_${eval_exp}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" || true
      exit 1
    fi
    sleep 5
  done
else
  echo "No LLM baseline: skipping embedding server step."
fi

# Build evaluation command
CMD=(
  "${PYTHON}" -u onpolicy/scripts/eval/eval_hanabi.py
  --env_name "${env}"
  --algorithm_name "${algo}"
  --experiment_name "${eval_exp}"
  --hanabi_name "${hanabi}"
  --num_agents "${num_agents}"
  --seed 1
  --n_training_threads 1
  --n_rollout_threads "${rollout_threads}"
  --n_eval_rollout_threads 10
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
  --use_eval
  --use_wandb
  --model_dir "${MODEL_DIR}"
)

if [ "${use_llm}" = "1" ]; then
  CMD+=(
    --use_llm
    --llm_model "${llm_model}"
    --llm_backend "${llm_backend}"
    --llm_vector_mode "${llm_vector_mode}"
    --llm_step_prob 1.0
    --qwen_embed_host "${EMBED_HOST}"
    --qwen_embed_port "${EMBED_PORT}"
  )
fi

echo "Starting Python Evaluation Loop at: $(date)"
"${CMD[@]}"

echo "Array Task Complete at: $(date)"
