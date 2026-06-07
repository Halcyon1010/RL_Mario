import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_rl.envs.wrappers import FrameStack, MarioObservation, MaxEpisodeSteps, SkipFrame


ACTION_SPACES = {
    "RIGHT_ONLY": RIGHT_ONLY,
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
}


def make_mario_env(env_id, config, max_episode_steps):
    env = gym_super_mario_bros.make(env_id)
    action_space_name = config.get("action_space", "RIGHT_ONLY")
    env = JoypadSpace(env, ACTION_SPACES[action_space_name])
    env = SkipFrame(env, skip=config.get("frame_skip", 4))
    env = MarioObservation(env, image_size=config.get("image_size", 84))
    env = FrameStack(env, num_stack=config.get("frame_stack", 4))
    env = MaxEpisodeSteps(env, max_episode_steps=max_episode_steps)
    return env
