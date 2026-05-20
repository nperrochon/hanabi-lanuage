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

# --- Variables ---
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"

# WARNING: with SubprocVecEnv, each rollout worker may load its own embedding model.
# If this OOMs or is slow, reduce this to 1 or 2.
rollout_threads=8

seed=1

# Qwen text embedding model, not Qwen instruct generation model.
llm_model="Qwen/Qwen3-Embedding-0.6B"
llm_backend="qwen_embedding"
llm_vector_mode="qwen_embedding"
llm_embedding_dim=1024
llm_step_prob=0.02

exp="qwen_embed_${rollout_threads}threads_step${llm_step_prob}_seed${seed}"

REPO="/data/class/mae93/nperroch/hanabi-lanuage"
LOG_DIR="${REPO}/logs"

mkdir -p "${LOG_DIR}"

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Repo: ${REPO}"
echo "Experiment: ${exp}"

# Make conda activate work in non-interactive SLURM shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- CUDA / GPU info ---
module avail cuda || true
module load cuda || true

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "CUDA_HOME=$CUDA_HOME"
  nvcc --version
else
  echo "nvcc not found; continuing as long as PyTorch sees CUDA."
fi

nvidia-smi || true

# --- Python env ---
conda activate marl

cd "${REPO}"

unset HF_ENDPOINT
export HF_HOME=/data/class/mae93/nperroch/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$HF_HOME"

echo "Checking Python / CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

echo "Checking Qwen embedding dependencies..."
python - <<'PY'
import sentence_transformers
import transformers
print("sentence_transformers:", sentence_transformers.__version__)
print("transformers:", transformers.__version__)
PY

echo "Pre-caching / verifying Qwen embedding model..."
python - <<PY
from sentence_transformers import SentenceTransformer
import torch

model_name = "${llm_model}"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_name} on {device}")
model = SentenceTransformer(model_name, device=device)
emb = model.encode(["test hanabi state"], convert_to_numpy=True, normalize_embeddings=True)
print("embedding shape:", emb.shape)

expected_dim = ${llm_embedding_dim}
assert emb.shape[1] == expected_dim, f"Expected dim {expected_dim}, got {emb.shape[1]}"
print("Qwen embedding model verified")
PY

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
  --llm_backend ${llm_backend} \
  --llm_vector_mode ${llm_vector_mode} \
  --llm_embedding_dim ${llm_embedding_dim} \
  --llm_step_prob ${llm_step_prob}

echo "Job ended at: $(date)"
