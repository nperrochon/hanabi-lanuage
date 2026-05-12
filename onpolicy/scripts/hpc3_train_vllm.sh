#!/bin/bash
#SBATCH --job-name=hanabi_mappo
#SBATCH --output=/data/class/mae93/nperroch/hanabi-lanuage/logs/hanabi_%j.out
#SBATCH --error=/data/class/mae93/nperroch/hanabi-lanuage/logs/hanabi_%j.err
#SBATCH -A royf_lab_gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:A100:1

# --- Variables ---
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
rollout_threads=8
llm_model="Qwen/Qwen2.5-14B-Instruct"
exp="single_run_${rollout_threads}threads_${llm_model}"
seed=1

REPO="/data/class/mae93/nperroch/hanabi-lanuage"
LOG_DIR="${REPO}/logs"

# Make conda activate work in non-interactive SLURM shell
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate vllm-server


module avail cuda
module load cuda
which nvcc
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version


hostname
nvidia-smi

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

# --- Environment Setup ---
export PATH="/data/class/mae93/nperroch/ollama-install/bin:$PATH"
echo "Starting vLLM server..."
conda activate vllm-server

MODEL="Qwen/Qwen2.5-14B-Instruct"
PORT=8002

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --generation-config vllm \
  --dtype float16 \
  --enable-prefix-caching \
  > "${LOG_DIR}/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM server..."
for i in {1..120}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null; then
    echo "vLLM ready after $i checks"
    break
  fi

  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process died while starting. Last server log lines:"
    tail -100 "$LOG_DIR/vllm_server_${SLURM_JOB_ID}.log"
    exit 1
  fi

  echo "Still waiting for vLLM... check $i"
  sleep 5
done

echo "Testing vLLM chat completion..."
curl -s http://127.0.0.1:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"Output only one integer. No words.\"},
      {\"role\": \"user\", \"content\": \"Choose one legal move index from 0 to 9. Answer:\"}
    ],
    \"max_tokens\": 2,
    \"temperature\": 0
  }"
echo ""

echo "Starting Python training..."
conda activate marl


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
  --llm_step_prob 0.02 \
  --llm_backend vllm \
  --llm_base_url http://127.0.0.1:${PORT}/v1 \
  --llm_vector_mode softmax


echo "Job ended at: $(date)"
