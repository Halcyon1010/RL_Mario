from collections import deque

import cv2
import gym
import numpy as np


class MaxEpisodeSteps(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.max_episode_steps = int(max_episode_steps)
        self.elapsed_steps = 0

    def reset(self, **kwargs):
        self.elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1
        timeout = self.elapsed_steps >= self.max_episode_steps
        if timeout:
            done = True
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = int(skip)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        skipped_frames = []
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            skipped_frames.append(np.ascontiguousarray(self.env.unwrapped.screen.copy()))
            if done:
                break
        info = dict(info)
        info["_skipped_rgb_frames"] = skipped_frames
        return obs, total_reward, done, info


class MarioObservation(gym.ObservationWrapper):
    def __init__(self, env, image_size):
        super().__init__(env)
        self.image_size = int(image_size)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size),
            dtype=np.uint8,
        )

    def observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = int(num_stack)
        self.frames = deque(maxlen=self.num_stack)
        low = np.repeat(env.observation_space.low[None, ...], self.num_stack, axis=0)
        high = np.repeat(env.observation_space.high[None, ...], self.num_stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)
