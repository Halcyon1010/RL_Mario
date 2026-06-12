from rllte.xplore.reward import E3B, ICM, NGU, RE3, RIDE, RND, Disagreement, PseudoCounts
from gymnasium import spaces


REWARD_MODULES = {
    "rnd": RND,
    "icm": ICM,
    "ride": RIDE,
    "ngu": NGU,
    "re3": RE3,
    "e3b": E3B,
    "pseudo_counts": PseudoCounts,
    "disagreement": Disagreement,
}


class SingleEnvRLLTEAdapter:
    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        self.single_observation_space = convert_space(env.observation_space)
        self.single_action_space = convert_space(env.action_space)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.unwrapped = self


def convert_space(space):
    if hasattr(space, "n"):
        return spaces.Discrete(space.n)
    if hasattr(space, "low") and hasattr(space, "high"):
        return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    raise NotImplementedError(f"Unsupported space for RLLTE adapter: {space}")


def build_intrinsic_reward(name, envs, device, **kwargs):
    key = name.lower()
    if key not in REWARD_MODULES:
        available = ", ".join(sorted(REWARD_MODULES))
        raise ValueError(f"Unknown intrinsic reward '{name}'. Available: {available}")
    return REWARD_MODULES[key](envs=envs, device=device, **kwargs)
