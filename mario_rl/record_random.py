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


def make_env(env_id: str):
    env = gym_super_mario_bros.make(env_id)
    return JoypadSpace(env, DEMO_MOVEMENT)


def freeze_frame(frame):
    """Copy emulator output into a stable RGB frame for video encoding."""
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")
    return np.ascontiguousarray(frame.copy())


def main():
    parser = argparse.ArgumentParser(description="Record a Mario rollout without training.")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="videos/mario_demo.mp4")
    parser.add_argument(
        "--mode",
        default="scripted",
        choices=["scripted", "random"],
        help="Use a scripted action timeline or random actions. This is only for video export, not learning.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id)
    state = env.reset()

    total_reward = 0.0
    max_x_pos = 0
    frames = [freeze_frame(state)]

    action_names = {
        0: "noop",
        1: "right",
        2: "right_jump",
        3: "right_run",
        4: "right_run_jump",
        5: "left",
        6: "left_jump",
    }

    scripted_actions = [
        ("stand_still", 0, 45),
        ("walk_right", 1, 120),
        ("jump_right", 2, 45),
        ("run_right", 3, 120),
        ("run_jump_right", 4, 80),
        ("pause", 0, 35),
        ("walk_left", 5, 70),
        ("jump_left", 6, 35),
        ("recover_right", 3, 140),
        ("finish_run_jump", 4, 120),
    ]

    if args.mode == "scripted":
        action_timeline = [
            action
            for _, action, duration in scripted_actions
            for _ in range(duration)
        ]
    else:
        random_steps = args.steps if args.steps is not None else 1200
        action_timeline = [None] * random_steps

    if args.steps is not None:
        action_timeline = action_timeline[: args.steps]

    action_counts = {name: 0 for name in action_names.values()}

    for action in action_timeline:
        action = env.action_space.sample() if action is None else action
        action_counts[action_names[action]] += 1
        state, reward, done, info = env.step(action)
        frames.append(freeze_frame(state))

        total_reward += float(reward)
        max_x_pos = max(max_x_pos, int(info.get("x_pos", 0)))

        if done:
            state = env.reset()
            frames.append(freeze_frame(state))

    env.close()

    imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=1)

    print("record=ok")
    print(f"output={output_path}")
    print(f"frames={len(frames)}")
    print(f"mode={args.mode}")
    print(f"total_reward={total_reward:.2f}")
    print(f"max_x_pos={max_x_pos}")
    print("action_counts=" + ", ".join(f"{key}:{value}" for key, value in action_counts.items() if value))


if __name__ == "__main__":
    main()
