#!/usr/bin/env python
import sys
from pathlib import Path

# This checkout must win over any other installed clone (e.g. another path on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import os
import wandb
import socket
import setproctitle
import numpy as np
import torch
from onpolicy.config import get_config
from onpolicy.envs.hanabi.Hanabi_Env import HanabiEnv
from onpolicy.envs.env_wrappers import ChooseSubprocVecEnv, ChooseDummyVecEnv
from onpolicy.runner.shared.hanabi_runner_forward import HanabiRunner as Runner
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("[MAIN] about to import transformers", flush=True)
from transformers import AutoTokenizer, AutoModel

print("[MAIN] imported transformers", flush=True)

"""Train script for Hanabi."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert (
                    all_args.num_agents > 1 and all_args.num_agents < 6
                ), "num_agents can be only between 2-5."
                env = HanabiEnv(all_args, (all_args.seed + rank * 1000))
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        print("Using ChooseDummyVecEnv with n_rollout_threads = 1")
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        print(
            "Using ChooseSubprocVecEnv with n_rollout_threads =",
            all_args.n_rollout_threads,
        )
        return ChooseSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        )


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert (
                    all_args.num_agents > 1 and all_args.num_agents < 6
                ), "num_agents can be only between 2-5."
                env = HanabiEnv(all_args, (all_args.seed * 50000 + rank * 10000))
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    # --env_name is already added by get_config(); only add Hanabi-specific args.
    parser.add_argument(
        "--hanabi_name",
        type=str,
        default="Hanabi-Very-Small",
        help="Which env to run on",
    )
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    # Only add LLM args if not already present (e.g. from config).
    if not any(a.dest == "use_llm" for a in parser._actions):
        parser.add_argument(
            "--use_llm",
            action="store_true",
            default=False,
            help="Append Ollama LLM action recommendation to observations (requires Ollama running).",
        )
    if not any(a.dest == "llm_model" for a in parser._actions):
        parser.add_argument(
            "--llm_model",
            type=str,
            default="qwen2:7b",
            help="Ollama model tag when --use_llm (e.g. qwen2:7b, llama3.2).",
        )

    if not any(a.dest == "llm_step_prob" for a in parser._actions):
        parser.add_argument(
            "--llm_step_prob",
            type=float,
            default=1.0,
            help="Probability of using LLM action recommendation at each step (0.0 = never, 1.0 = always).",
        )

    all_args = parser.parse_known_args(args)[0]
    if not any(x == "--env_name" or x.startswith("--env_name=") for x in args):
        all_args.env_name = "Hanabi"
    # Default wandb entity to user's workspace when not explicitly set
    if not any(x == "--user_name" or x.startswith("--user_name=") for x in args):
        all_args.user_name = "nperroch-uci"
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.hanabi_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            group=all_args.hanabi_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Make it explicit whether LLM is used (default: False)
    use_llm = getattr(all_args, "use_llm", False)
    print("use_llm: {}".format(use_llm))
    if use_llm:
        print("llm_model: {}".format(getattr(all_args, "llm_model", "qwen2:7b")))
        # With an LLM in the loop, each env step is expensive. Shorten episode_length
        # so we get more frequent policy updates for the same num_env_steps.
        # (Keep it at least 50 to preserve some temporal structure.)
        old_len = all_args.episode_length
        all_args.episode_length = max(50, old_len // 2)
        print(
            "Using shorter episode_length with LLM: {} -> {}".format(
                old_len, all_args.episode_length
            )
        )

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
