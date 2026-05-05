from turtle import up
import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.save_every_x_updates = self.all_args.save_every_x_updates

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        # policy network
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            device=self.device,
        )

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        # If a full checkpoint was loaded before trainer creation, restore trainer state now
        if hasattr(self, "_resume_checkpoint") and not self.use_eval:
            ckpt = self._resume_checkpoint

            # Policy optimizers
            if "actor_optimizer_state_dict" in ckpt:
                if hasattr(self.trainer.policy, "actor_optimizer"):
                    self.trainer.policy.actor_optimizer.load_state_dict(
                        ckpt["actor_optimizer_state_dict"]
                    )
                elif hasattr(self.trainer, "actor_optimizer"):
                    self.trainer.actor_optimizer.load_state_dict(
                        ckpt["actor_optimizer_state_dict"]
                    )
                print("Restored actor optimizer state")

            if "critic_optimizer_state_dict" in ckpt:
                if hasattr(self.trainer.policy, "critic_optimizer"):
                    self.trainer.policy.critic_optimizer.load_state_dict(
                        ckpt["critic_optimizer_state_dict"]
                    )
                elif hasattr(self.trainer, "critic_optimizer"):
                    self.trainer.critic_optimizer.load_state_dict(
                        ckpt["critic_optimizer_state_dict"]
                    )
                print("Restored critic optimizer state")

            # Value normalizer / PopArt
            if "value_normalizer_state_dict" in ckpt:
                if hasattr(self.trainer, "value_normalizer") and self.trainer.value_normalizer is not None:
                    if hasattr(self.trainer.value_normalizer, "load_state_dict"):
                        self.trainer.value_normalizer.load_state_dict(
                            ckpt["value_normalizer_state_dict"]
                        )
                        print("Restored value normalizer state")
            elif "value_normalizer_obj" in ckpt:
                self.trainer.value_normalizer = ckpt["value_normalizer_obj"]
                print("Restored serialized value normalizer object")

        # buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, update_num=0, final=False):
        """Save full training state for real resume."""
        suffix = "" if final else f"_ep{update_num}"
        ckpt_path = (
            os.path.join(self.save_dir, "checkpoint.pt")
            if final
            else os.path.join(self.save_dir, f"checkpoint{suffix}.pt")
        )

        checkpoint = {
            "actor_state_dict": self.trainer.policy.actor.state_dict(),
            "critic_state_dict": self.trainer.policy.critic.state_dict(),
        }

        # Optimizers: save if present
        if hasattr(self.trainer.policy, "actor_optimizer"):
            checkpoint["actor_optimizer_state_dict"] = (
                self.trainer.policy.actor_optimizer.state_dict()
            )
        if hasattr(self.trainer.policy, "critic_optimizer"):
            checkpoint["critic_optimizer_state_dict"] = (
                self.trainer.policy.critic_optimizer.state_dict()
            )

        # Some codebases keep optimizers on trainer, not policy
        if hasattr(self.trainer, "actor_optimizer"):
            checkpoint["actor_optimizer_state_dict"] = (
                self.trainer.actor_optimizer.state_dict()
            )
        if hasattr(self.trainer, "critic_optimizer"):
            checkpoint["critic_optimizer_state_dict"] = (
                self.trainer.critic_optimizer.state_dict()
            )

        # Value normalizer / PopArt
        if hasattr(self.trainer, "value_normalizer") and self.trainer.value_normalizer is not None:
            if hasattr(self.trainer.value_normalizer, "state_dict"):
                checkpoint["value_normalizer_state_dict"] = (
                    self.trainer.value_normalizer.state_dict()
                )
            else:
                # fallback for custom objects without state_dict
                checkpoint["value_normalizer_obj"] = self.trainer.value_normalizer

        # Useful metadata
        checkpoint["all_args"] = vars(self.all_args)
        checkpoint["hidden_size"] = self.hidden_size
        checkpoint["recurrent_N"] = self.recurrent_N
        checkpoint["update_num"] = update_num

        torch.save(checkpoint, ckpt_path)
        print(f"Saved full checkpoint to {ckpt_path}")

        # Optional: keep old files too for eval compatibility
        torch.save(
            self.trainer.policy.actor.state_dict(),
            os.path.join(self.save_dir, "actor.pt" if final else f"actor_ep{update_num}.pt"),
        )
        torch.save(
            self.trainer.policy.critic.state_dict(),
            os.path.join(self.save_dir, "critic.pt" if final else f"critic_ep{update_num}.pt"),
        )

    def restore(self, model_dir):
        """Restore full training state if available; otherwise fall back to actor/critic only."""
        print(f"Restoring model from {model_dir}")

        checkpoint_path = os.path.join(model_dir, "checkpoint.pt")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            actor_missing, actor_unexpected = self.policy.actor.load_state_dict(
                checkpoint["actor_state_dict"], strict=False
            )
            print("Actor missing:", actor_missing)
            print("Actor unexpected:", actor_unexpected)

            if not self.all_args.use_render:
                critic_missing, critic_unexpected = self.policy.critic.load_state_dict(
                    checkpoint["critic_state_dict"], strict=False
                )
                print("Critic missing:", critic_missing)
                print("Critic unexpected:", critic_unexpected)

            print("Loaded actor/critic weights from checkpoint.pt")

            self._resume_checkpoint = checkpoint
            return

        # Fallback: old style actor/critic only
        actor_path = os.path.join(model_dir, "actor.pt")
        critic_path = os.path.join(model_dir, "critic.pt")

        policy_actor_state_dict = torch.load(actor_path, map_location=self.device)
        actor_missing, actor_unexpected = self.policy.actor.load_state_dict(
            policy_actor_state_dict, strict=False
        )
        print("Actor missing:", actor_missing)
        print("Actor unexpected:", actor_unexpected)

        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(critic_path, map_location=self.device)
            critic_missing, critic_unexpected = self.policy.critic.load_state_dict(
                policy_critic_state_dict, strict=False
            )
            print("Critic missing:", critic_missing)
            print("Critic unexpected:", critic_unexpected)

        print("Loaded legacy actor.pt / critic.pt only")

    def log_train(self, train_infos, total_num_steps):
            """
            Log training info.
            :param train_infos: (dict) information about training update.
            :param total_num_steps: (int) total number of training env steps.
            """
            for k, v in train_infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
