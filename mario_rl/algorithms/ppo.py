import time
import argparse
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from mario_rl.envs.make_env import make_mario_env
from mario_rl.logging.csv_logger import CsvLogger, save_json, save_yaml
from mario_rl.rewards.rllte_intrinsic import SingleEnvRLLTEAdapter, build_intrinsic_reward
from mario_rl.utils.config import load_config
from mario_rl.utils.seed import set_seed


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        channels = obs_shape[0]
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, obs):
        return self.critic(self.encoder(obs / 255.0))

    def get_action_and_value(self, obs, action=None):
        hidden = self.encoder(obs / 255.0)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)

    def get_greedy_action(self, obs):
        hidden = self.encoder(obs / 255.0)
        logits = self.actor(hidden)
        return torch.argmax(logits, dim=-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def obs_to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def reward_for_update(reward, train_config):
    if not train_config.get("reward_clip", False):
        return float(reward)
    return float(
        np.clip(
            reward,
            float(train_config.get("reward_clip_low", -1.0)),
            float(train_config.get("reward_clip_high", 1.0)),
        )
    )


def build_intrinsic_module(config, env, device):
    reward_config = config.get("reward", {}).get("intrinsic", {})
    if not reward_config.get("enabled", False):
        return None

    kwargs = {
        key: value
        for key, value in reward_config.items()
        if key not in {"enabled", "type"}
    }
    kwargs.setdefault("obs_norm_type", "none")
    return build_intrinsic_reward(
        reward_config.get("type", "rnd"),
        envs=SingleEnvRLLTEAdapter(env),
        device=str(device),
        **kwargs,
    )


def compute_intrinsic_rollout(intrinsic_reward, observations, actions, ext_rewards, dones, next_observations):
    if intrinsic_reward is None:
        return torch.zeros_like(ext_rewards)

    samples = {
        "observations": observations.unsqueeze(1),
        "actions": actions.view(-1, 1),
        "rewards": ext_rewards.view(-1, 1),
        "terminateds": dones.view(-1, 1),
        "truncateds": torch.zeros_like(dones).view(-1, 1),
        "next_observations": next_observations.unsqueeze(1),
    }
    with torch.enable_grad():
        rewards = intrinsic_reward.compute(samples, sync=True).view(-1)
    rewards = torch.nan_to_num(rewards.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return rewards


def watch_intrinsic_step(intrinsic_reward, observation, action, ext_reward, done, next_observation, device):
    if intrinsic_reward is None:
        return
    try:
        intrinsic_reward.watch(
            observations=observation.unsqueeze(0),
            actions=action.view(1),
            rewards=torch.as_tensor([ext_reward], dtype=torch.float32, device=device),
            terminateds=torch.as_tensor([done], dtype=torch.float32, device=device),
            truncateds=torch.zeros(1, dtype=torch.float32, device=device),
            next_observations=next_observation.unsqueeze(0),
        )
    except NotImplementedError:
        return


EVAL_FIELDNAMES = [
    "split",
    "eval_mode",
    "global_step",
    "global_decision_step",
    "approx_global_env_frame_step",
    "env_id",
    "episode",
    "ext_reward",
    "int_reward",
    "total_reward",
    "decision_steps",
    "approx_env_frame_steps",
    "final_x_pos",
    "max_x_pos",
    "mean_x_pos",
    "timeout",
    "max_episode_steps",
    "frame_skip",
]


def summarize_eval(results):
    if not results:
        return {
            "mean_ext_reward": 0.0,
            "mean_int_reward": 0.0,
            "mean_total_reward": 0.0,
            "mean_max_x_pos": 0.0,
            "mean_decision_steps": 0.0,
        }
    return {
        "mean_ext_reward": float(np.mean([row["ext_reward"] for row in results])),
        "mean_int_reward": float(np.mean([row["int_reward"] for row in results])),
        "mean_total_reward": float(np.mean([row["total_reward"] for row in results])),
        "mean_max_x_pos": float(np.mean([row["max_x_pos"] for row in results])),
        "mean_decision_steps": float(np.mean([row["decision_steps"] for row in results])),
    }


def evaluate(agent, config, device, global_step, run_path, logger, split, env_ids, episodes, max_episode_steps, eval_mode, wandb_run=None):
    env_config = config["env"]
    agent.eval()
    results = []

    for env_id in env_ids:
        for episode_idx in range(int(episodes)):
            env = make_mario_env(env_id, env_config, max_episode_steps)
            obs = env.reset()
            done = False
            episode_step = 0
            ext_reward = 0.0
            int_reward = 0.0
            final_x_pos = 0
            max_x_pos = 0
            x_pos_sum = 0.0
            timeout = False

            while not done:
                with torch.no_grad():
                    obs_tensor = obs_to_tensor(obs, device)
                    if eval_mode == "greedy":
                        action = agent.get_greedy_action(obs_tensor)
                    elif eval_mode == "sample":
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    else:
                        raise ValueError(f"Unknown eval_mode: {eval_mode}")
                obs, reward, done, info = env.step(int(action.item()))
                episode_step += 1
                ext_reward += float(reward)
                int_reward += 0.0
                final_x_pos = int(info.get("x_pos", 0))
                max_x_pos = max(max_x_pos, final_x_pos)
                x_pos_sum += final_x_pos
                timeout = bool(info.get("TimeLimit.truncated", False))

            env.close()
            mean_x_pos = x_pos_sum / max(episode_step, 1)
            total_reward = ext_reward + int_reward
            row = {
                "split": split,
                "eval_mode": eval_mode,
                "global_step": global_step,
                "global_decision_step": global_step,
                "approx_global_env_frame_step": global_step * env_config["frame_skip"],
                "env_id": env_id,
                "episode": episode_idx,
                "ext_reward": ext_reward,
                "int_reward": int_reward,
                "total_reward": total_reward,
                "decision_steps": episode_step,
                "approx_env_frame_steps": episode_step * env_config["frame_skip"],
                "final_x_pos": final_x_pos,
                "max_x_pos": max_x_pos,
                "mean_x_pos": mean_x_pos,
                "timeout": int(timeout),
                "max_episode_steps": max_episode_steps,
                "frame_skip": env_config["frame_skip"],
            }
            logger.write(row)
            wandb_log(
                wandb_run,
                {
                    f"{split}/ext_reward": ext_reward,
                    f"{split}/int_reward": int_reward,
                    f"{split}/total_reward": total_reward,
                    f"{split}/decision_steps": episode_step,
                    f"{split}/final_x_pos": final_x_pos,
                    f"{split}/max_x_pos": max_x_pos,
                    f"{split}/mean_x_pos": mean_x_pos,
                    f"{split}/timeout": int(timeout),
                    f"{split}/{eval_mode}/ext_reward": ext_reward,
                    f"{split}/{eval_mode}/int_reward": int_reward,
                    f"{split}/{eval_mode}/total_reward": total_reward,
                    f"{split}/{eval_mode}/decision_steps": episode_step,
                    f"{split}/{eval_mode}/max_x_pos": max_x_pos,
                },
                step=global_step,
            )
            results.append(row)

    agent.train()
    summary = summarize_eval(results)
    save_json(run_path / f"latest_{split}_{eval_mode}.json", {"summary": summary, "episodes": results})
    wandb_log(
        wandb_run,
        {
            f"{split}/mean_ext_reward": summary["mean_ext_reward"],
            f"{split}/mean_int_reward": summary["mean_int_reward"],
            f"{split}/mean_total_reward": summary["mean_total_reward"],
            f"{split}/mean_max_x_pos": summary["mean_max_x_pos"],
            f"{split}/mean_decision_steps": summary["mean_decision_steps"],
            f"{split}/{eval_mode}/mean_ext_reward": summary["mean_ext_reward"],
            f"{split}/{eval_mode}/mean_int_reward": summary["mean_int_reward"],
            f"{split}/{eval_mode}/mean_total_reward": summary["mean_total_reward"],
            f"{split}/{eval_mode}/mean_max_x_pos": summary["mean_max_x_pos"],
            f"{split}/{eval_mode}/mean_decision_steps": summary["mean_decision_steps"],
        },
        step=global_step,
    )
    return results, summary


def apply_overrides(config, overrides):
    if overrides.total_timesteps is not None:
        config["training"]["total_timesteps"] = overrides.total_timesteps
    if overrides.eval_interval is not None:
        config["training"]["eval_interval"] = overrides.eval_interval
    if overrides.eval_episodes is not None:
        config["training"]["eval_episodes"] = overrides.eval_episodes
    if overrides.test_episodes is not None:
        config["training"]["test_episodes"] = overrides.test_episodes
    if overrides.save_interval is not None:
        config["training"]["save_interval"] = overrides.save_interval
    if overrides.device is not None:
        config["training"]["device"] = overrides.device
    if overrides.run_dir is not None:
        config["experiment"]["run_dir"] = overrides.run_dir
    if overrides.seed is not None:
        config["experiment"]["seed"] = overrides.seed
    if overrides.experiment_name is not None:
        config["experiment"]["name"] = overrides.experiment_name
    config.setdefault("wandb", {})
    if overrides.track_wandb:
        config["wandb"]["enabled"] = True
    if overrides.no_wandb:
        config["wandb"]["enabled"] = False
    if overrides.wandb_project is not None:
        config["wandb"]["project"] = overrides.wandb_project
    if overrides.wandb_entity is not None:
        config["wandb"]["entity"] = overrides.wandb_entity
    if overrides.wandb_mode is not None:
        config["wandb"]["mode"] = overrides.wandb_mode
    return config


def build_arg_parser(default_config=None):
    parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros.")
    parser.add_argument("--config", default=default_config, help="Path to a YAML config file.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--track-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    return parser


def init_wandb(config, run_name):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None
    secrets_path = Path(wandb_config.get("secrets_path", "secrets/wandb.json"))
    if not os.environ.get("WANDB_API_KEY") and secrets_path.exists():
        with secrets_path.open("r", encoding="utf-8-sig") as handle:
            secrets = json.load(handle)
        api_key = secrets.get("api_key")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
    import wandb

    return wandb.init(
        project=wandb_config.get("project", "mario-intrinsic-ppo"),
        entity=wandb_config.get("entity"),
        name=run_name,
        config=config,
        mode=wandb_config.get("mode", "online"),
        sync_tensorboard=False,
    )


def wandb_log(wandb_run, metrics, step):
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def train(config_path, overrides=None):
    config = load_config(config_path)
    if overrides is not None:
        config = apply_overrides(config, overrides)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    train_config = config["training"]
    env_config = config["env"]
    device = select_device(train_config.get("device", "auto"))

    run_name = f"{config['experiment']['name']}_seed{seed}_{int(time.time())}"
    run_path = Path(config["experiment"].get("run_dir", "runs")) / run_name
    checkpoint_path = run_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    save_yaml(run_path / "config.yaml", config)
    wandb_run = init_wandb(config, run_name)

    env = make_mario_env(env_config["train_env_id"], env_config, env_config["train_max_episode_steps"])
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = ActorCritic(obs_shape, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(train_config["learning_rate"]), eps=1e-5)
    intrinsic_reward = build_intrinsic_module(config, env, device)

    rollout_steps = int(train_config["rollout_steps"])
    obs_buf = torch.zeros((rollout_steps,) + obs_shape, device=device)
    next_obs_buf = torch.zeros((rollout_steps,) + obs_shape, device=device)
    actions_buf = torch.zeros((rollout_steps,), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((rollout_steps,), device=device)
    ext_rewards_buf = torch.zeros((rollout_steps,), device=device)
    rewards_buf = torch.zeros((rollout_steps,), device=device)
    dones_buf = torch.zeros((rollout_steps,), device=device)
    done_after_buf = torch.zeros((rollout_steps,), device=device)
    values_buf = torch.zeros((rollout_steps,), device=device)

    train_logger = CsvLogger(
        run_path / "train_metrics.csv",
        [
            "global_step",
            "global_decision_step",
            "approx_global_env_frame_step",
            "episode",
            "episode_decision_steps",
            "episode_approx_env_frame_steps",
            "episode_ext_reward",
            "episode_int_reward",
            "episode_total_reward",
            "episode_final_x_pos",
            "episode_max_x_pos",
            "episode_mean_x_pos",
            "visited_bins",
            "episode_timeout",
            "train_total_timesteps",
            "train_max_episode_steps",
            "frame_skip",
        ],
    )
    update_logger = CsvLogger(
        run_path / "update_metrics.csv",
        [
            "global_step",
            "global_decision_step",
            "approx_global_env_frame_step",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "learning_rate",
            "rollout_action_0",
            "rollout_action_1",
            "rollout_action_2",
            "rollout_action_3",
            "rollout_action_4",
            "rollout_action_5",
            "rollout_action_6",
            "rollout_action_7",
            "rollout_action_8",
            "rollout_action_9",
            "rollout_action_10",
            "rollout_action_11",
            "dominant_action_frac",
            "reward_clip",
            "target_kl",
            "rollout_int_reward_mean",
            "rollout_int_reward_sum",
        ],
    )
    val_logger = CsvLogger(run_path / "val_metrics.csv", EVAL_FIELDNAMES)
    test_logger = CsvLogger(run_path / "test_metrics.csv", EVAL_FIELDNAMES)

    obs = env.reset()
    global_step = 0
    episode = 0
    episode_step = 0
    episode_ext_reward = 0.0
    episode_int_reward = 0.0
    episode_total_reward = 0.0
    episode_final_x_pos = 0
    episode_max_x_pos = 0
    episode_x_pos_sum = 0.0
    visited_bins = set()
    next_done = torch.zeros((), device=device)
    recent_x_positions = deque(maxlen=100)

    total_timesteps = int(train_config["total_timesteps"])
    next_eval_step = int(train_config["eval_interval"])
    best_val_total_reward = -float("inf")
    best_summary = None
    eval_modes = train_config.get("eval_modes", ["greedy"])
    best_model_eval_mode = train_config.get("best_model_eval_mode", eval_modes[0])

    while global_step < total_timesteps:
        rollout_len = 0
        for step in range(rollout_steps):
            rollout_len += 1
            global_step += 1
            obs_buf[step] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_buf[step].unsqueeze(0))
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            action_item = int(action.item())
            next_obs, ext_reward, done, info = env.step(action_item)
            next_obs_buf[step] = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            ext_rewards_buf[step] = float(ext_reward)
            done_after_buf[step] = float(done)
            watch_intrinsic_step(
                intrinsic_reward,
                obs_buf[step],
                actions_buf[step],
                float(ext_reward),
                bool(done),
                next_obs_buf[step],
                device,
            )
            int_reward = 0.0

            episode_step += 1
            episode_ext_reward += float(ext_reward)
            episode_total_reward += float(ext_reward)
            x_pos = int(info.get("x_pos", 0))
            episode_final_x_pos = x_pos
            episode_max_x_pos = max(episode_max_x_pos, x_pos)
            episode_x_pos_sum += x_pos
            recent_x_positions.append(x_pos)
            visited_bins.add(x_pos // 50)

            next_done = torch.tensor(float(done), device=device)
            obs = next_obs

            if done:
                episode_mean_x_pos = episode_x_pos_sum / max(episode_step, 1)
                train_logger.write(
                    train_row := {
                        "global_step": global_step,
                        "global_decision_step": global_step,
                        "approx_global_env_frame_step": global_step * env_config["frame_skip"],
                        "episode": episode,
                        "episode_decision_steps": episode_step,
                        "episode_approx_env_frame_steps": episode_step * env_config["frame_skip"],
                        "episode_ext_reward": episode_ext_reward,
                        "episode_int_reward": episode_int_reward,
                        "episode_total_reward": episode_total_reward,
                        "episode_final_x_pos": episode_final_x_pos,
                        "episode_max_x_pos": episode_max_x_pos,
                        "episode_mean_x_pos": episode_mean_x_pos,
                        "visited_bins": len(visited_bins),
                        "episode_timeout": int(bool(info.get("TimeLimit.truncated", False))),
                        "train_total_timesteps": total_timesteps,
                        "train_max_episode_steps": env_config["train_max_episode_steps"],
                        "frame_skip": env_config["frame_skip"],
                    }
                )
                wandb_log(
                    wandb_run,
                    {
                        "train/episode_ext_reward": episode_ext_reward,
                        "train/episode_int_reward": episode_int_reward,
                        "train/episode_total_reward": episode_total_reward,
                        "train/episode_decision_steps": episode_step,
                        "train/episode_max_x_pos": episode_max_x_pos,
                        "train/episode_final_x_pos": episode_final_x_pos,
                        "train/episode_mean_x_pos": episode_mean_x_pos,
                        "train/visited_bins": len(visited_bins),
                        "train/episode_timeout": int(bool(info.get("TimeLimit.truncated", False))),
                    },
                    step=global_step,
                )
                obs = env.reset()
                episode += 1
                episode_step = 0
                episode_ext_reward = 0.0
                episode_int_reward = 0.0
                episode_total_reward = 0.0
                episode_final_x_pos = 0
                episode_max_x_pos = 0
                episode_x_pos_sum = 0.0
                visited_bins = set()
                next_done = torch.zeros((), device=device)

            if global_step >= next_eval_step:
                val_summaries = {}
                for eval_mode in eval_modes:
                    _, mode_summary = evaluate(
                        agent,
                        config,
                        device,
                        global_step,
                        run_path,
                        val_logger,
                        split="val",
                        env_ids=env_config.get("val_env_ids", env_config.get("eval_env_ids", [env_config["train_env_id"]])),
                        episodes=int(train_config["eval_episodes"]),
                        max_episode_steps=env_config.get("val_max_episode_steps", env_config.get("eval_max_episode_steps")),
                        eval_mode=eval_mode,
                        wandb_run=wandb_run,
                    )
                    val_summaries[eval_mode] = mode_summary
                val_summary = val_summaries[best_model_eval_mode]
                if val_summary["mean_total_reward"] > best_val_total_reward:
                    best_val_total_reward = val_summary["mean_total_reward"]
                    torch.save(agent.state_dict(), checkpoint_path / "best.pt")
                    test_summaries = {}
                    test_episode_rows = {}
                    for eval_mode in eval_modes:
                        test_results, mode_test_summary = evaluate(
                            agent,
                            config,
                            device,
                            global_step,
                            run_path,
                            test_logger,
                            split="test",
                            env_ids=env_config.get("test_env_ids", env_config.get("val_env_ids", env_config.get("eval_env_ids", [env_config["train_env_id"]]))),
                            episodes=int(train_config.get("test_episodes", train_config["eval_episodes"])),
                            max_episode_steps=env_config.get("test_max_episode_steps", env_config.get("val_max_episode_steps", env_config.get("eval_max_episode_steps"))),
                            eval_mode=eval_mode,
                            wandb_run=wandb_run,
                        )
                        test_summaries[eval_mode] = mode_test_summary
                        test_episode_rows[eval_mode] = test_results
                    best_summary = {
                        "global_step": global_step,
                        "best_model_eval_mode": best_model_eval_mode,
                        "val": val_summaries,
                        "test": test_summaries,
                        "test_episodes": test_episode_rows,
                    }
                    save_json(run_path / "best_summary.json", best_summary)
                next_eval_step += int(train_config["eval_interval"])

            if global_step >= total_timesteps:
                break

        rollout_obs_for_intrinsic = obs_buf[:rollout_len]
        rollout_next_obs_for_intrinsic = next_obs_buf[:rollout_len]
        rollout_ext_rewards = ext_rewards_buf[:rollout_len]
        rollout_actions_for_intrinsic = actions_buf[:rollout_len]
        rollout_done_after = done_after_buf[:rollout_len]
        rollout_int_rewards_tensor = compute_intrinsic_rollout(
            intrinsic_reward,
            rollout_obs_for_intrinsic,
            rollout_actions_for_intrinsic,
            rollout_ext_rewards,
            rollout_done_after,
            rollout_next_obs_for_intrinsic,
        )
        rollout_total_rewards = rollout_ext_rewards + rollout_int_rewards_tensor
        if train_config.get("reward_clip", False):
            rollout_rewards_for_update = torch.clamp(
                rollout_total_rewards,
                float(train_config.get("reward_clip_low", -1.0)),
                float(train_config.get("reward_clip_high", 1.0)),
            )
        else:
            rollout_rewards_for_update = rollout_total_rewards
        rewards_buf[:rollout_len] = rollout_rewards_for_update
        rollout_int_rewards = rollout_int_rewards_tensor.detach().cpu().numpy().tolist()

        with torch.no_grad():
            next_value = agent.get_value(obs_to_tensor(obs, device)).reshape(1)
            rollout_rewards = rewards_buf[:rollout_len]
            rollout_dones = dones_buf[:rollout_len]
            rollout_values = values_buf[:rollout_len]
            advantages = torch.zeros_like(rollout_rewards)
            last_gae = 0
            for t in reversed(range(rollout_len)):
                if t == rollout_len - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - rollout_dones[t + 1]
                    next_values = rollout_values[t + 1]
                delta = rollout_rewards[t] + train_config["gamma"] * next_values * next_nonterminal - rollout_values[t]
                advantages[t] = last_gae = delta + train_config["gamma"] * train_config["gae_lambda"] * next_nonterminal * last_gae
            returns = advantages + rollout_values

        rollout_obs = obs_buf[:rollout_len]
        rollout_actions = actions_buf[:rollout_len]
        rollout_logprobs = logprobs_buf[:rollout_len]
        batch_inds = np.arange(rollout_len)
        clip_fractions = []
        stop_updates = False
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        approx_kl_value = 0.0

        for _ in range(int(train_config["update_epochs"])):
            np.random.shuffle(batch_inds)
            for start in range(0, rollout_len, int(train_config["minibatch_size"])):
                end = start + int(train_config["minibatch_size"])
                mb_inds = batch_inds[start:end]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(rollout_obs[mb_inds], rollout_actions[mb_inds])
                logratio = new_logprob - rollout_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fractions.append(((ratio - 1.0).abs() > train_config["clip_coef"]).float().mean().item())

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - train_config["clip_coef"],
                    1 + train_config["clip_coef"],
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                new_value = new_value.view(-1)
                value_loss = 0.5 * ((new_value - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - train_config["entropy_coef"] * entropy_loss + train_config["value_coef"] * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), train_config["max_grad_norm"])
                optimizer.step()

                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                entropy_value = float(entropy_loss.item())
                approx_kl_value = float(approx_kl.item())

                target_kl = train_config.get("target_kl")
                if target_kl is not None and approx_kl_value > float(target_kl):
                    stop_updates = True
                    break
            if stop_updates:
                break

        action_counts = torch.bincount(rollout_actions.detach().cpu(), minlength=num_actions).numpy()
        dominant_action_frac = float(action_counts.max() / max(action_counts.sum(), 1))

        update_row = {
                "global_step": global_step,
                "global_decision_step": global_step,
                "approx_global_env_frame_step": global_step * env_config["frame_skip"],
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl_value,
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "rollout_action_0": int(action_counts[0]) if num_actions > 0 else 0,
                "rollout_action_1": int(action_counts[1]) if num_actions > 1 else 0,
                "rollout_action_2": int(action_counts[2]) if num_actions > 2 else 0,
                "rollout_action_3": int(action_counts[3]) if num_actions > 3 else 0,
                "rollout_action_4": int(action_counts[4]) if num_actions > 4 else 0,
                "rollout_action_5": int(action_counts[5]) if num_actions > 5 else 0,
                "rollout_action_6": int(action_counts[6]) if num_actions > 6 else 0,
                "rollout_action_7": int(action_counts[7]) if num_actions > 7 else 0,
                "rollout_action_8": int(action_counts[8]) if num_actions > 8 else 0,
                "rollout_action_9": int(action_counts[9]) if num_actions > 9 else 0,
                "rollout_action_10": int(action_counts[10]) if num_actions > 10 else 0,
                "rollout_action_11": int(action_counts[11]) if num_actions > 11 else 0,
                "dominant_action_frac": dominant_action_frac,
                "reward_clip": int(bool(train_config.get("reward_clip", False))),
                "target_kl": train_config.get("target_kl"),
                "rollout_int_reward_mean": float(np.mean(rollout_int_rewards)) if rollout_int_rewards else 0.0,
                "rollout_int_reward_sum": float(np.sum(rollout_int_rewards)) if rollout_int_rewards else 0.0,
            }
        update_logger.write(update_row)
        wandb_log(
            wandb_run,
            {
                "update/policy_loss": policy_loss_value,
                "update/value_loss": value_loss_value,
                "update/entropy": entropy_value,
                "update/approx_kl": approx_kl_value,
                "update/clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "update/dominant_action_frac": dominant_action_frac,
                "update/learning_rate": optimizer.param_groups[0]["lr"],
                "update/rollout_int_reward_mean": float(np.mean(rollout_int_rewards)) if rollout_int_rewards else 0.0,
                "update/rollout_int_reward_sum": float(np.sum(rollout_int_rewards)) if rollout_int_rewards else 0.0,
            },
            step=global_step,
        )

        if global_step % int(train_config["save_interval"]) == 0 or global_step >= total_timesteps:
            torch.save(agent.state_dict(), checkpoint_path / f"ppo_step_{global_step}.pt")
            torch.save(agent.state_dict(), checkpoint_path / "last.pt")

    final_val_summaries = {}
    for eval_mode in eval_modes:
        _, mode_final_val_summary = evaluate(
            agent,
            config,
            device,
            global_step,
            run_path,
            val_logger,
            split="val",
            env_ids=env_config.get("val_env_ids", env_config.get("eval_env_ids", [env_config["train_env_id"]])),
            episodes=int(train_config["eval_episodes"]),
            max_episode_steps=env_config.get("val_max_episode_steps", env_config.get("eval_max_episode_steps")),
            eval_mode=eval_mode,
            wandb_run=wandb_run,
        )
        final_val_summaries[eval_mode] = mode_final_val_summary
    final_val_summary = final_val_summaries[best_model_eval_mode]
    if final_val_summary["mean_total_reward"] > best_val_total_reward:
        best_val_total_reward = final_val_summary["mean_total_reward"]
        torch.save(agent.state_dict(), checkpoint_path / "best.pt")
        test_summaries = {}
        test_episode_rows = {}
        for eval_mode in eval_modes:
            test_results, mode_test_summary = evaluate(
                agent,
                config,
                device,
                global_step,
                run_path,
                test_logger,
                split="test",
                env_ids=env_config.get("test_env_ids", env_config.get("val_env_ids", env_config.get("eval_env_ids", [env_config["train_env_id"]]))),
                episodes=int(train_config.get("test_episodes", train_config["eval_episodes"])),
                max_episode_steps=env_config.get("test_max_episode_steps", env_config.get("val_max_episode_steps", env_config.get("eval_max_episode_steps"))),
                eval_mode=eval_mode,
                wandb_run=wandb_run,
            )
            test_summaries[eval_mode] = mode_test_summary
            test_episode_rows[eval_mode] = test_results
        best_summary = {
            "global_step": global_step,
            "best_model_eval_mode": best_model_eval_mode,
            "val": final_val_summaries,
            "test": test_summaries,
            "test_episodes": test_episode_rows,
        }
        save_json(run_path / "best_summary.json", best_summary)
    torch.save(agent.state_dict(), checkpoint_path / "last.pt")
    env.close()
    train_logger.close()
    update_logger.close()
    val_logger.close()
    test_logger.close()
    save_json(
        run_path / "summary.json",
        {
            "run_name": run_name,
            "global_step": global_step,
            "episodes": episode,
            "device": str(device),
            "obs_shape": obs_shape,
            "num_actions": num_actions,
            "action_space": env_config.get("action_space", "RIGHT_ONLY"),
            "frame_skip": env_config["frame_skip"],
            "best_val_total_reward": best_val_total_reward,
            "best_summary": best_summary,
        },
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "final_global_step": global_step,
                "episodes": episode,
                "device": str(device),
                "num_actions": num_actions,
                "action_space": env_config.get("action_space", "RIGHT_ONLY"),
            }
        )
        wandb_run.finish()
    print(f"train=ok run_dir={run_path}")


def main(default_config=None):
    parser = build_arg_parser(default_config=default_config)
    args = parser.parse_args()
    if args.config is None:
        parser.error("--config is required")
    train(args.config, overrides=args)
