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


def main():
    parser = argparse.ArgumentParser(description="Record the best video from many checkpoint rollouts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mode", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--save-best-by", choices=["max_x_pos", "total_reward"], default="max_x_pos")
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = None

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
            score = summary[args.save_best_by]
            improved = best is None or score > best[args.save_best_by]
            if improved:
                best = summary
                imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=1)
            print(
                "episode_summary="
                f"episode:{episode_idx},"
                f"score:{score:.2f},"
                f"max_x_pos:{max_x_pos},"
                f"ext_reward:{ext_reward:.2f},"
                f"decision_steps:{decision_steps},"
                f"best_episode:{best['episode']},"
                f"best_max_x_pos:{best['max_x_pos']}"
            )
    finally:
        env.close()

    print("best_video=ok")
    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"output={output_path}")
    print(f"episodes={args.episodes}")
    print(f"mode={args.mode}")
    print(f"save_best_by={args.save_best_by}")
    print(f"best_episode={best['episode']}")
    print(f"best_decision_steps={best['decision_steps']}")
    print(f"best_ext_reward={best['ext_reward']:.2f}")
    print(f"best_final_x_pos={best['final_x_pos']}")
    print(f"best_max_x_pos={best['max_x_pos']}")
    print(f"best_mean_x_pos={best['mean_x_pos']:.2f}")
    print("best_action_counts=" + ",".join(f"{idx}:{count}" for idx, count in enumerate(best["action_counts"].tolist())))


if __name__ == "__main__":
    main()
