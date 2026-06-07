import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from mario_rl.algorithms.ppo import ActorCritic, select_device
from mario_rl.envs.make_env import make_mario_env
from mario_rl.evaluate_checkpoint import choose_action, freeze_frame, get_rgb_frame
from mario_rl.utils.config import load_config
from mario_rl.utils.seed import set_seed


def write_video(path: Path, frames, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def main():
    parser = argparse.ArgumentParser(
        description="Record reward-best and x-position-best videos from the same rollout samples."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-reward", required=True)
    parser.add_argument("--output-x", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--mode", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    env_config = config["env"]
    env_id = args.env_id or env_config.get("train_env_id", "SuperMarioBros-1-1-v0")
    max_episode_steps = args.max_episode_steps or env_config.get(
        "test_max_episode_steps",
        env_config.get("eval_max_episode_steps", 3000),
    )
    device = select_device(args.device)

    env = make_mario_env(env_id, env_config, max_episode_steps)
    agent = ActorCritic(env.observation_space.shape, env.action_space.n).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    output_reward = Path(args.output_reward)
    output_x = Path(args.output_x)
    best_reward = None
    best_x = None

    try:
        for episode_idx in range(args.episodes):
            obs = env.reset()
            frames = [get_rgb_frame(env)]
            done = False
            decision_steps = 0
            ext_reward = 0.0
            final_x_pos = 0
            max_x_pos = 0
            x_pos_sum = 0.0
            action_counts = np.zeros(env.action_space.n, dtype=np.int64)

            while not done:
                action = choose_action(agent, obs, device, args.mode)
                action_counts[action] += 1
                obs, reward, done, info = env.step(action)
                decision_steps += 1
                ext_reward += float(reward)
                final_x_pos = int(info.get("x_pos", 0))
                max_x_pos = max(max_x_pos, final_x_pos)
                x_pos_sum += final_x_pos
                if "_skipped_rgb_frames" in info:
                    frames.extend(freeze_frame(frame) for frame in info["_skipped_rgb_frames"])
                else:
                    frames.append(get_rgb_frame(env))

            summary = {
                "episode": episode_idx,
                "decision_steps": decision_steps,
                "ext_reward": ext_reward,
                "total_reward": ext_reward,
                "final_x_pos": final_x_pos,
                "max_x_pos": max_x_pos,
                "mean_x_pos": x_pos_sum / max(decision_steps, 1),
                "action_counts": action_counts,
            }

            if best_reward is None or summary["total_reward"] > best_reward["total_reward"]:
                best_reward = summary
                write_video(output_reward, frames, args.fps)
            if best_x is None or summary["max_x_pos"] > best_x["max_x_pos"]:
                best_x = summary
                write_video(output_x, frames, args.fps)

            print(
                "episode_summary="
                f"episode:{episode_idx},"
                f"total_reward:{summary['total_reward']:.2f},"
                f"max_x_pos:{summary['max_x_pos']},"
                f"decision_steps:{summary['decision_steps']},"
                f"best_reward_episode:{best_reward['episode']},"
                f"best_total_reward:{best_reward['total_reward']:.2f},"
                f"best_reward_max_x_pos:{best_reward['max_x_pos']},"
                f"best_x_episode:{best_x['episode']},"
                f"best_x_pos:{best_x['max_x_pos']},"
                f"best_x_total_reward:{best_x['total_reward']:.2f}"
            )
    finally:
        env.close()

    print("dual_best_videos=ok")
    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"env_id={env_id}")
    print(f"episodes={args.episodes}")
    print(f"mode={args.mode}")
    print(f"output_reward={output_reward}")
    print(f"reward_best_episode={best_reward['episode']}")
    print(f"reward_best_total_reward={best_reward['total_reward']:.2f}")
    print(f"reward_best_max_x_pos={best_reward['max_x_pos']}")
    print(f"reward_best_final_x_pos={best_reward['final_x_pos']}")
    print(f"reward_best_decision_steps={best_reward['decision_steps']}")
    print(f"reward_best_mean_x_pos={best_reward['mean_x_pos']:.2f}")
    print(
        "reward_best_action_counts="
        + ",".join(f"{idx}:{count}" for idx, count in enumerate(best_reward["action_counts"].tolist()))
    )
    print(f"output_x={output_x}")
    print(f"x_best_episode={best_x['episode']}")
    print(f"x_best_total_reward={best_x['total_reward']:.2f}")
    print(f"x_best_max_x_pos={best_x['max_x_pos']}")
    print(f"x_best_final_x_pos={best_x['final_x_pos']}")
    print(f"x_best_decision_steps={best_x['decision_steps']}")
    print(f"x_best_mean_x_pos={best_x['mean_x_pos']:.2f}")
    print(
        "x_best_action_counts="
        + ",".join(f"{idx}:{count}" for idx, count in enumerate(best_x["action_counts"].tolist()))
    )


if __name__ == "__main__":
    main()
