"""FishRL: a small reproducible testbed for exploration and reward shaping."""

from .environment import FishEnv, RewardConfig
from .agent import DQNAgent, AgentConfig

__all__ = ["FishEnv", "RewardConfig", "DQNAgent", "AgentConfig"]
