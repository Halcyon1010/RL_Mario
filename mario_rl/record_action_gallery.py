import argparse
from pathlib import Path

import gym_super_mario_bros
import imageio.v2 as imageio
import numpy as np
from nes_py.wrappers import JoypadSpace


DEMO_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["left"],
    ["left", "A"],
]

ACTION_IDS = {
    "noop": 0,
    "right": 1,
    "right_jump": 2,
    "right_run": 3,
    "right_run_jump": 4,
    "left": 5,
    "left_jump": 6,
}


def make_env(env_id: str):
    env = gym_super_mario_bros.make(env_id)
    return JoypadSpace(env, DEMO_MOVEMENT)


def freeze_frame(frame):
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")
    return np.ascontiguousarray(frame.copy())


def warm_up(env, steps: int):
    state = env.reset()
    for _ in range(steps):
        state, _, done, _ = env.step(ACTION_IDS["right"])
        if done:
            state = env.reset()
    return state


def record_action(env_id: str, action_name: str, steps: int, warmup_steps: int, fps: int, output_dir: Path):
    env = make_env(env_id)
    state = warm_up(env, warmup_steps)

    action = ACTION_IDS[action_name]
    frames = [freeze_frame(state)]
    total_reward = 0.0
    max_x_pos = 0

    for _ in range(steps):
        state, reward, done, info = env.step(action)
        frames.append(freeze_frame(state))
        total_reward += float(reward)
        max_x_pos = max(max_x_pos, int(info.get("x_pos", 0)))

        if done:
            state = env.reset()
            frames.append(freeze_frame(state))

    env.close()

    mp4_path = output_dir / f"{action_name}.mp4"
    gif_path = output_dir / f"{action_name}.gif"

    imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=1)
    imageio.mimsave(gif_path, frames, fps=fps)

    return {
        "action": action_name,
        "mp4": str(mp4_path),
        "gif": str(gif_path),
        "frames": len(frames),
        "total_reward": total_reward,
        "max_x_pos": max_x_pos,
    }


def main():
    parser = argparse.ArgumentParser(description="Record separate Mario action demo videos and GIFs.")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--output-dir", default="videos/action_gallery")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["noop", "right", "right_jump", "right_run", "right_run_jump", "left", "left_jump"],
        choices=list(ACTION_IDS),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("gallery=start")
    for action_name in args.actions:
        result = record_action(
            env_id=args.env_id,
            action_name=action_name,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            fps=args.fps,
            output_dir=output_dir,
        )
        print(
            "gallery_item=ok "
            f"action={result['action']} "
            f"mp4={result['mp4']} "
            f"gif={result['gif']} "
            f"frames={result['frames']} "
            f"max_x_pos={result['max_x_pos']} "
            f"total_reward={result['total_reward']:.2f}"
        )
    print("gallery=ok")


if __name__ == "__main__":
    main()
