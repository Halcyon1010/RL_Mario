import argparse

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace


def make_env(env_id: str):
    env = gym_super_mario_bros.make(env_id)
    return JoypadSpace(env, RIGHT_ONLY)


def main():
    parser = argparse.ArgumentParser(description="Check the Mario RL Docker environment.")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    env = make_env(args.env_id)
    state = env.reset()

    print(f"env_id={args.env_id}")
    print(f"action_space={env.action_space}")
    print(f"initial_observation_shape={getattr(state, 'shape', None)}")

    total_reward = 0.0
    max_x_pos = 0

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        max_x_pos = max(max_x_pos, int(info.get("x_pos", 0)))

        print(
            f"step={step:03d} action={action} reward={reward:.2f} "
            f"x_pos={info.get('x_pos')} done={done}"
        )

        if done:
            state = env.reset()

    env.close()

    print("check=ok")
    print(f"total_reward={total_reward:.2f}")
    print(f"max_x_pos={max_x_pos}")


if __name__ == "__main__":
    main()
