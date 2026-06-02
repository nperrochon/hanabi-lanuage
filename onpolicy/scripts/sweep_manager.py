#!/usr/bin/env python3
"""
Hanabi Multi-Agent RL Multi-Model Grid Search & Sweep Orchestrator
Automates job resource allocation across an expanded model spectrum (33M to 7B parameters).
Constrained to 8 threads / 1 mini-batch for complete optimization stability.
"""

import os
import subprocess
import time

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

# FIXED CONTROL PARAMETERS FOR STABLE CONVERGENCE
THREADS = 8
MINI_BATCH = 1


def generate_dynamic_slurm(job_name, cmd_args, log_base, tier):
    """Dynamically scales cluster resources based on model memory footprints."""
    if tier == "large":
        gpu_request = (
            "#SBATCH --gres=gpu:a30:1"  # Allocates safe VRAM nodes for 7B models
        )
        cpus_per_task = 16
        time_limit = "#SBATCH --time=3-00:00:00"
    elif tier == "medium":
        gpu_request = "#SBATCH --gres=gpu:1"
        cpus_per_task = 8
        time_limit = "#SBATCH --time=3-00:00:00"
    else:  # small tier
        gpu_request = "#SBATCH --gres=gpu:1"
        cpus_per_task = 4
        time_limit = "#SBATCH --time=1-12:00:00"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
{gpu_request}
{time_limit}
#SBATCH --output={log_base}.out
#SBATCH --error={log_base}.err

module load cuda/12.1
source activate marl

echo "Starting Sweep Run: {job_name} ({tier} model) at $(date)"
python -u train_hanabi_forward.py {cmd_args}
echo "Finished Sweep Run at $(date)"
"""
    return script


def main():
    os.makedirs("sweep_scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    job_counter = 0
    base_port = 9300

    for model, backend, mode, tier in MODELS_AND_BACKENDS:
        for prob in PROBABILITIES:
            for seed in SEEDS:
                job_counter += 1
                current_port = base_port + (job_counter % 500)

                # Format clean strings for WandB dashboard sorting
                model_short = (
                    model.split("/")[-1]
                    .replace("-Instruct", "")
                    .replace("-Embedding", "")
                    .replace("-L6-v2", "")
                )
                exp_name = f"sweep_{model_short}_{mode}_p{prob}_t{THREADS}"
                job_uid = f"hn_sw_{job_counter:03d}"

                args_list = [
                    f"--env_name Hanabi",
                    f"--hanabi_name Hanabi-Full",
                    f"--algorithm_name mappo",
                    f"--experiment_name {exp_name}",
                    f"--num_agents 2",
                    f"--seed {seed}",
                    f"--n_rollout_threads {THREADS}",
                    f"--num_mini_batch {MINI_BATCH}",
                    f"--use_llm",
                    f"--llm_model {model}",
                    f"--llm_backend {backend}",
                    f"--llm_vector_mode {mode}",
                    f"--llm_step_prob {prob}",
                    f"--use_wandb",
                    f"--save_every_x_updates 200",  # Protects your runs with metric checkpoints
                    f"--log_interval 5",
                ]

                if backend == "qwen_embedding":
                    args_list.append(f"--qwen_embed_host 127.0.0.1")
                    args_list.append(f"--qwen_embed_port {current_port}")
                else:
                    args_list.append(
                        f"--llm_base_url http://127.0.0.1:{current_port}/v1"
                    )

                cmd_args = " ".join(args_list)
                log_base = f"logs/{job_uid}_{exp_name}"

                script_content = generate_dynamic_slurm(
                    job_uid, cmd_args, log_base, tier
                )
                script_path = f"sweep_scripts/run_{job_uid}.sh"

                with open(script_path, "w") as f:
                    f.write(script_content)

                print(
                    f"[{job_uid.upper()} - {tier.upper()} TIER] Launching configuration: {exp_name} on Port {current_port}..."
                )
                subprocess.run(["sbatch", script_path])
                time.sleep(1.5)

    print(
        f"\n[SWEEP ARRANGEMENT SETUP COMPLETE] Successfully automated {job_counter} controlled trajectories."
    )


if __name__ == "__main__":
    main()
