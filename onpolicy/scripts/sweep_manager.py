#!/usr/bin/env python3
"""
Hanabi Multi-Agent RL Multi-Model Grid Search & Sweep Orchestrator
Tailored specifically for UCI HPC3 cluster guidelines and constraints.
Constrained to 8 threads / 1 mini-batch for complete optimization stability.
"""

import os
import subprocess
import time

# ==============================================================================
# ⚠️ UCI HPC3 CONFIGURATION ADJUSTMENTS
# ==============================================================================
CHARGE_ACCOUNT = "royf_lab_gpu"  # Extracted from your sample hpc3 configuration
REPO_PATH = "/data/class/mae93/nperroch/hanabi-lanuage"  # Your absolute cluster path
HF_CACHE_DIR = (
    "/data/class/mae93/nperroch/huggingface"  # Your HuggingFace mirror target
)
# ==============================================================================

# --- EXPANDED ALL-MODEL SWEEP MATRIX ---
MODELS_AND_BACKENDS = [
    # (model_name, backend, vector_mode, parameter_tier)
    # === TINY / LOW-DIMENSIONAL ENCODERS ===
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "qwen_embedding",
        "qwen_embedding",
        "small",
    ),
    ("BAAI/bge-small-en-v1.5", "qwen_embedding", "qwen_embedding", "small"),
    (
        "sentence-transformers/all-mpnet-base-v2",
        "qwen_embedding",
        "qwen_embedding",
        "small",
    ),
    ("Qwen/Qwen2.5-0.5B-Instruct", "vllm", "softmax", "small"),
    # === MEDIUM / STATE-OF-THE-ART ENCODERS ===
    ("Qwen/Qwen3-Embedding-0.6B", "qwen_embedding", "qwen_embedding", "medium"),
    ("BAAI/bge-large-en-v1.5", "qwen_embedding", "qwen_embedding", "medium"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "vllm", "softmax", "medium"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "vllm", "analysis_embedding", "medium"),
    # === LARGE / GENERATIVE COGNITIVE MODELS ===
    ("Qwen/Qwen2.5-7B-Instruct", "vllm", "softmax", "large"),
    ("Qwen/Qwen2.5-7B-Instruct", "vllm", "one_hot", "large"),
    ("Qwen/Qwen2.5-7B-Instruct", "vllm", "analysis_embedding", "large"),
]

PROBABILITIES = [0.02, 0.10, 0.20, 0.50]
SEEDS = [1]

# FIXED STABLE REINFORCEMENT LEARNING HYPERPARAMETERS
THREADS = 8
MINI_BATCH = 1


def generate_hpc3_slurm(
    job_uid, exp_name, model, backend, mode, tier, port, prob, seed
):
    """Generates an absolute mirror of your verified HPC3 multi-process layout script."""

    # Set safe, predictable hardware resource requirements per model size
    if tier == "large":
        gpu_request = "#SBATCH --gres=gpu:A100:1"  # Matches your sample high-performance node type
        cpus_per_task = 16
        mem_request = "#SBATCH --mem=128G"
        time_limit = "#SBATCH --time=3-00:00:00"
    elif tier == "medium":
        gpu_request = "#SBATCH --gres=gpu:1"
        cpus_per_task = 8
        mem_request = "#SBATCH --mem=64G"
        time_limit = "#SBATCH --time=3-00:00:00"
    else:  # small tier
        gpu_request = "#SBATCH --gres=gpu:1"
        cpus_per_task = 4
        mem_request = "#SBATCH --mem=32G"
        time_limit = "#SBATCH --time=1-12:00:00"

    # Define background orchestration strings dynamically
    if backend == "qwen_embedding":
        server_cmd = (
            f"python -u onpolicy/envs/hanabi/qwen_embedding_server.py \\\n"
            f'  --model "{model}" \\\n'
            f'  --host "127.0.0.1" \\\n'
            f'  --port "{port}" \\\n'
            f"  --device cuda \\\n"
            f'  > "${{LOG_DIR}}/qwen_embed_server_${{SLURM_JOB_ID}}.log" 2>&1 &'
        )
        health_url = f"http://127.0.0.1:{port}/health"
        health_grep = "grep -q '\"ok\": true'"
    else:  # vllm server deployment
        server_cmd = (
            f"python -m vllm.entrypoints.openai.api_server \\\n"
            f'  --model "{model}" \\\n'
            f'  --host "127.0.0.1" \\\n'
            f'  --port "{port}" \\\n'
            f"  --disable-log-requests \\\n"
            f"  --max-model-len 2048 \\\n"
            f"  --enable-prefix-caching \\\n"
            f'  > "${{LOG_DIR}}/vllm_server_${{SLURM_JOB_ID}}.log" 2>&1 &'
        )
        health_url = f"http://127.0.0.1:{port}/v1/models"
        health_grep = 'grep -q \'"object": "list"\''

    # Dynamically inject custom networking arguments into the main training process
    network_args = (
        f"--qwen_embed_host 127.0.0.1 --qwen_embed_port {port}"
        if backend == "qwen_embedding"
        else f"--llm_base_url http://127.0.0.1:{port}/v1"
    )

    script = f"""#!/bin/bash
#SBATCH --job-name={job_uid}_{tier}
#SBATCH --output={REPO_PATH}/logs/{job_uid}_{exp_name}_%j.out
#SBATCH --error={REPO_PATH}/logs/{job_uid}_{exp_name}_%j.err
#SBATCH -A {CHARGE_ACCOUNT}
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
{mem_request}
{time_limit}
{gpu_request}

set -euo pipefail

echo "Job started at: $(date)"
echo "Host Node: $(hostname)"
echo "Experiment Target: {exp_name}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marl

cd "{REPO_PATH}"

unset HF_ENDPOINT
export HF_HOME={HF_CACHE_DIR}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

nvidia-smi || true

echo "Starting isolated node model backend server..."
{server_cmd}

EMBED_PID=$!

cleanup() {{
  echo "Cleaning up local background model server PID=${{EMBED_PID}}"
  kill "${{EMBED_PID}}" 2>/dev/null || true
}}
trap cleanup EXIT

echo "Executing server link verification loop..."
for i in {{1..120}}; do
  if curl -s "{health_url}" | {health_grep}; then
    echo "Model backend server successfully verified after ${{i}} validation intervals."
    break
  fi

  if ! kill -0 "${{EMBED_PID}}" 2>/dev/null; then
    echo "CRITICAL: Background server failed structurally during boot phase. Diagnostic tail:"
    tail -50 "${{LOG_DIR}}/"*_"${{SLURM_JOB_ID}}".log
    exit 1
  fi

  echo "Server response pending... retry step ${{i}}"
  sleep 5
done

echo "Executing main RL training execution loop at: $(date)"

python -u onpolicy/scripts/train/train_hanabi_forward.py \\
  --env_name Hanabi \\
  --algorithm_name mappo \\
  --experiment_name {exp_name} \\
  --hanabi_name Hanabi-Full \\
  --num_agents 2 \\
  --seed {seed} \\
  --n_training_threads 1 \\
  --n_rollout_threads {THREADS} \\
  --num_mini_batch {MINI_BATCH} \\
  --episode_length 100 \\
  --num_env_steps 1000000000 \\
  --ppo_epoch 10 \\
  --gain 0.01 \\
  --lr 7e-4 \\
  --critic_lr 1e-3 \\
  --clip_param 0.1 \\
  --hidden_size 512 \\
  --layer_N 2 \\
  --entropy_coef 0.015 \\
  --log_interval 5 \\
  --use_llm \\
  --llm_model "{model}" \\
  --llm_backend {backend} \\
  --llm_vector_mode {mode} \\
  --llm_step_prob {prob} \\
  --use_wandb \\
  --save_every_x_updates 200 \\
  {network_args}

echo "Job execution loop cleanly exited at: $(date)"
"""
    return script


def main():
    os.makedirs(f"{REPO_PATH}/sweep_scripts", exist_ok=True)
    os.makedirs(f"{REPO_PATH}/logs", exist_ok=True)

    job_counter = 0
    # Base port configuration offset to guarantee absolute zero overlap risks inside Slurm
    base_port = 12400

    for model, backend, mode, tier in MODELS_AND_BACKENDS:
        for prob in PROBABILITIES:
            for seed in SEEDS:
                job_counter += 1
                current_port = base_port + job_counter

                # Sanitize naming labels for clean metric comparisons on WandB
                model_short = (
                    model.split("/")[-1]
                    .replace("-Instruct", "")
                    .replace("-Embedding", "")
                    .replace("-L6-v2", "")
                )
                exp_name = f"sw_{model_short}_{mode}_p{prob}"
                job_uid = f"h3_{job_counter:03d}"

                script_content = generate_hpc3_slurm(
                    job_uid,
                    exp_name,
                    model,
                    backend,
                    mode,
                    tier,
                    current_port,
                    prob,
                    seed,
                )

                script_path = f"{REPO_PATH}/sweep_scripts/run_{job_uid}.sh"
                with open(script_path, "w") as f:
                    f.write(script_content)

                print(
                    f"[{job_uid.upper()} - {tier.upper()} TIER] Building safe shell execution wrapper: {exp_name} on Port {current_port}..."
                )
                subprocess.run(["sbatch", script_path])
                time.sleep(1.0)

    print(
        f"\n[PIPELINE READY] {job_counter} verified cluster run scripts generated and enqueued."
    )


if __name__ == "__main__":
    main()
