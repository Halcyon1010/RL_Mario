import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from mario_rl.algorithms.ppo import ActorCritic, obs_to_tensor, select_device
from mario_rl.envs.make_env import make_mario_env
from mario_rl.utils.config import load_config
from mario_rl.utils.seed import set_seed


def freeze_frame(frame):
    frame = np.asarray(frame, dtype=np.uint8)
    return np.ascontiguousarray(frame.copy())


def get_rgb_frame(env):
    try:
        frame = env.render(mode="rgb_array")
        if frame is not None:
            return freeze_frame(frame)
    except TypeError:
        pass
    return freeze_frame(env.unwrapped.screen)


def choose_action(agent, obs, device, mode):
    obs_tensor = obs_to_tensor(obs, device)
    with torch.no_grad():
        if mode == "greedy":
            action = agent.get_greedy_action(obs_tensor)
        elif mode == "sample":
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return int(action.item())


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint and optionally record video.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="videos/checkpoint_eval.mp4")
    parser.add_argument("--mode", default="sample", choices=["sample", "greedy"])
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--playback",
        default="realtime",
        choices=["realtime", "decision"],
        help="realtime compensates for frame skip; decision writes one decision step as one video frame.",
    )
    parser.add_argument(
        "--record-skipped-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record every internal frame repeated by frame skip instead of only one frame per decision step.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save-best-by", default="max_x_pos", choices=["max_x_pos", "total_reward"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-video", action="store_true")
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

    best = None
    episode_summaries = []

    for episode_idx in range(args.episodes):
        obs = env.reset()
        frames = []
        if not args.no_video:
            frames.append(get_rgb_frame(env))

        done = False
        decision_steps = 0
        ext_reward = 0.0
        int_reward = 0.0
        final_x_pos = 0
        max_x_pos = 0
        x_pos_sum = 0.0
        timeout = False
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
            timeout = bool(info.get("TimeLimit.truncated", False))

            if not args.no_video:
                if args.record_skipped_frames and "_skipped_rgb_frames" in info:
                    frames.extend(freeze_frame(frame) for frame in info["_skipped_rgb_frames"])
                else:
                    frames.append(get_rgb_frame(env))

        total_reward = ext_reward + int_reward
        mean_x_pos = x_pos_sum / max(decision_steps, 1)
        summary = {
            "episode": episode_idx,
            "decision_steps": decision_steps,
            "ext_reward": ext_reward,
            "int_reward": int_reward,
            "total_reward": total_reward,
            "final_x_pos": final_x_pos,
            "max_x_pos": max_x_pos,
            "mean_x_pos": mean_x_pos,
            "timeout": int(timeout),
            "action_counts": action_counts,
            "frames": frames,
        }
        episode_summaries.append(summary)

        if best is None or summary[args.save_best_by] > best[args.save_best_by]:
            best = summary

    env.close()

    if not args.no_video:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_fps = args.fps
        if args.playback == "realtime" and not args.record_skipped_frames:
            video_fps = max(1, round(args.fps / env_config.get("frame_skip", 4)))
        imageio.mimsave(output_path, best["frames"], fps=video_fps, macro_block_size=1)
    else:
        output_path = None

    decision_steps = best["decision_steps"]
    ext_reward = best["ext_reward"]
    int_reward = best["int_reward"]
    total_reward = best["total_reward"]
    final_x_pos = best["final_x_pos"]
    max_x_pos = best["max_x_pos"]
    mean_x_pos = best["mean_x_pos"]
    timeout = bool(best["timeout"])
    action_counts = best["action_counts"]
    print("checkpoint_eval=ok")
    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"output={output_path}")
    print(f"video_fps={video_fps if not args.no_video else None}")
    print(f"playback={args.playback}")
    print(f"record_skipped_frames={int(args.record_skipped_frames)}")
    print(f"mode={args.mode}")
    print(f"episodes={args.episodes}")
    print(f"saved_episode={best['episode']}")
    print(f"save_best_by={args.save_best_by}")
    print(f"env_id={env_id}")
    print(f"decision_steps={decision_steps}")
    print(f"approx_env_frame_steps={decision_steps * env_config.get('frame_skip', 4)}")
    print(f"ext_reward={ext_reward:.2f}")
    print(f"int_reward={int_reward:.2f}")
    print(f"total_reward={total_reward:.2f}")
    print(f"final_x_pos={final_x_pos}")
    print(f"max_x_pos={max_x_pos}")
    print(f"mean_x_pos={mean_x_pos:.2f}")
    print(f"timeout={int(timeout)}")
    print("action_counts=" + ",".join(f"{idx}:{count}" for idx, count in enumerate(action_counts.tolist())))
    for summary in episode_summaries:
        print(
            "episode_summary="
            f"episode:{summary['episode']},"
            f"total_reward:{summary['total_reward']:.2f},"
            f"ext_reward:{summary['ext_reward']:.2f},"
            f"max_x_pos:{summary['max_x_pos']},"
            f"decision_steps:{summary['decision_steps']}"
        )


if __name__ == "__main__":
    main()
