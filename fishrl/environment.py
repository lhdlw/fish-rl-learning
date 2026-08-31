from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    name: str = "sparse"
    eat: float = 5.0
    death: float = -5.0
    approach_small: float = 0.0
    danger: float = 0.0
    idle: float = 0.0

    @staticmethod
    def preset(name: str) -> "RewardConfig":
        presets = {
            "sparse": RewardConfig(name="sparse"),
            "approach": RewardConfig(name="approach", approach_small=0.15),
            "approach_danger": RewardConfig(
                name="approach_danger", approach_small=0.15, danger=0.5
            ),
            "full": RewardConfig(
                name="full", approach_small=0.15, danger=0.5, idle=0.1
            ),
        }
        if name not in presets:
            raise ValueError(f"unknown reward preset: {name}")
        return presets[name]


class FishEnv:
    """Seeded vector environment with task metrics separated from training reward.

    Observation contains the controlled fish (x, y, size) and the six nearest
    NPC fish (relative x/y, relative size, signed horizontal velocity). Sorting
    by distance gives nearest-neighbour slots, not persistent fish identities.
    Remaining horizon is included for the fixed-horizon task (28 features).
    """

    width, height = 800.0, 600.0
    max_npc, min_npc = 6, 4
    action_dim = 4
    state_dim = 4 + max_npc * 4
    version = 'finite-horizon-v2'

    def __init__(
        self,
        reward: RewardConfig | str = "sparse",
        seed: int = 0,
        max_steps: int = 1000,
        big_fish_prob: float = 0.4,
    ):
        self.reward_cfg = RewardConfig.preset(reward) if isinstance(reward, str) else reward
        if max_steps <= 0:
            raise ValueError('max_steps must be positive')
        self.max_steps = max_steps
        self.big_fish_prob = big_fish_prob
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.npc: list[list[float]] = []
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)
        self.x, self.y, self.size = 400.0, 300.0, 30.0
        self.steps = self.fish_eaten = 0
        self.deaths = 0
        self._ended = False
        self.npc = []
        for _ in range(4):
            self._spawn()
        return self._observation()

    def _spawn(self) -> None:
        left = bool(self.rng.integers(0, 2))
        size = 55.0 if self.rng.random() < self.big_fish_prob else 25.0
        x = 0.0 if left else self.width
        y = float(self.rng.integers(50, int(self.height - 50) + 1))
        velocity = float(self.rng.uniform(1.5, 3.0)) * (1.0 if left else -1.0)
        self.npc.append([x, y, size, velocity])

    def _observation(self) -> np.ndarray:
        own = [self.x / self.width, self.y / self.height, self.size / 100.0]
        ordered = sorted(
            self.npc,
            key=lambda f: (self.x - f[0]) ** 2 + (self.y - f[1]) ** 2,
        )[: self.max_npc]
        features: list[float] = []
        for fish in ordered:
            features.extend([
                (fish[0] - self.x) / self.width,
                (fish[1] - self.y) / self.height,
                (fish[2] - self.size) / 100.0,
                fish[3] / 3.0,
            ])
        features.extend([0.0] * (4 * (self.max_npc - len(ordered))))
        remaining = max(0.0, 1.0 - self.steps / self.max_steps)
        return np.asarray(own + features + [remaining], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if action not in range(self.action_dim):
            raise ValueError("action must be in {0,1,2,3}")
        if self._ended:
            raise RuntimeError('Episode ended; call reset before step')
        self.steps += 1
        before = (self.x, self.y)
        if action == 0:
            self.y -= 6.0
        elif action == 1:
            self.y += 6.0
        elif action == 2:
            self.x -= 6.0
        else:
            self.x += 6.0
        self.x = float(np.clip(self.x, 20.0, self.width - 20.0))
        self.y = float(np.clip(self.y, 20.0, self.height - 20.0))

        for fish in self.npc:
            fish[0] += fish[3]
        self.npc = [f for f in self.npc if -50.0 < f[0] < self.width + 50.0]
        while len(self.npc) < self.min_npc:
            self._spawn()

        components = {"eat": 0.0, "death": 0.0, "approach": 0.0,
                      "danger": 0.0, "idle": 0.0}
        done = False
        eaten = []
        for i, fish in enumerate(self.npc):
            distance = float(np.hypot(self.x - fish[0], self.y - fish[1]))
            if distance < (self.size + fish[2]) / 2.2:
                if fish[2] < self.size:
                    components["eat"] += self.reward_cfg.eat
                    self.fish_eaten += 1
                    self.size += 2.0
                    eaten.append(i)
                else:
                    components["death"] += self.reward_cfg.death
                    self.deaths = 1
                    done = True
                    break
        for i in reversed(eaten):
            del self.npc[i]

        if not done:
            small_d = [np.hypot(self.x-f[0], self.y-f[1]) for f in self.npc if f[2] < self.size]
            big_d = [np.hypot(self.x-f[0], self.y-f[1]) for f in self.npc if f[2] >= self.size]
            if small_d:
                components["approach"] = self.reward_cfg.approach_small * float(
                    np.exp(-min(small_d) / 120.0)
                )
            if big_d and min(big_d) < 130.0:
                components["danger"] = -self.reward_cfg.danger * float(
                    np.exp(-min(big_d) / 60.0)
                )
            if (self.x, self.y) == before:
                components["idle"] = -self.reward_cfg.idle

        death = done
        time_limit = self.steps >= self.max_steps
        # Both death and the observed finite horizon end future task rewards.
        # An external training budget cutoff does not set this terminal mask.
        done = death or time_limit
        self._ended = done
        training_reward = float(sum(components.values()))
        # A reward-independent task return, used only for fair evaluation.
        objective_return = 5.0 * self.fish_eaten - 5.0 * self.deaths
        info = {
            **components,
            "fish_eaten": float(self.fish_eaten),
            "death": float(self.deaths),
            "objective_return": objective_return,
            "terminated": float(done),
            "time_limit": float(time_limit),
            "truncated": 0.0,
        }
        return self._observation(), training_reward, done, info
