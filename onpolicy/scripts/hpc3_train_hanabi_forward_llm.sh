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

# --- Variables ---
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=8
llm_model="qwen2:7b"
exp="single_run_${rollout_threads}threads_${llm_model}"
seed=1

REPO="/data/class/mae93/nperroch/hanabi-lanuage"
LOG_DIR="${REPO}/logs"

conda activate vllm-server

hostname
nvidia-smi

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

# --- Environment Setup ---
export PATH="/data/class/mae93/nperroch/ollama-install/bin:$PATH"
export OLLAMA_MODELS="/data/class/mae93/nperroch/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
ulimit -n 22222

mkdir -p "$LOG_DIR"
mkdir -p "$OLLAMA_MODELS"
cd "$REPO" || exit 1

# --- Start Ollama Server ---
echo "Starting Ollama server..."
ollama serve > "${LOG_DIR}/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

# Ensure Ollama dies when the job ends
cleanup() {
  echo "Cleaning up Ollama server (PID: ${OLLAMA_PID})..."
  kill "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for Ollama to be responsive
echo "Waiting for Ollama server to respond..."
for i in {1..60}; do
  if ollama list >/dev/null 2>&1; then
    echo "Ollama is ready after $i iterations"
    break
  fi
  sleep 2
done

# --- Prepare Model ---
echo "Pulling model ${llm_model}..."
ollama pull "${llm_model}"

echo "Warming up model (loading into VRAM)..."
# This ensures the model is loaded before Python starts 8 concurrent workers
ollama run "${llm_model}" "Say 1" > /dev/null

# --- Run Training ---
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
  --llm_step_prob 0.02

echo "Job ended at: $(date)"
