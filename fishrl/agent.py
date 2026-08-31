from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn


@dataclass
class AgentConfig:
    architecture: str = "dueling"
    exploration: str = "epsilon"
    lr: float = 1e-4
    gamma: float = 0.93
    batch_size: int = 64
    buffer_size: int = 80_000
    warmup: int = 500
    target_update: int = 300
    epsilon_start: float = 0.9
    epsilon_end: float = 0.02
    decay_steps: int = 30_000
    temperature_start: float = 1.0
    temperature_end: float = 0.05


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, architecture: str):
        super().__init__()
        self.architecture = architecture
        self.features = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.ReLU())
        if architecture == "dueling":
            self.value = nn.Linear(64, 1)
            self.advantage = nn.Linear(64, action_dim)
        elif architecture == "vanilla":
            self.head = nn.Linear(64, action_dim)
        else:
            raise ValueError(f"unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        if self.architecture == "vanilla":
            return self.head(z)
        value, advantage = self.value(z), self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def add(self, *transition) -> None:
        self.data.append(transition)

    def sample(self, n: int):
        batch = random.sample(self.data, n)
        s, a, r, ns, d = zip(*batch)
        return (np.asarray(s, np.float32), np.asarray(a, np.int64),
                np.asarray(r, np.float32), np.asarray(ns, np.float32),
                np.asarray(d, np.float32))

    def __len__(self):
        return len(self.data)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig,
                 seed: int = 0, device: str | None = None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cfg, self.action_dim = config, action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(state_dim, action_dim, config.architecture).to(self.device)
        self.target = QNetwork(state_dim, action_dim, config.architecture).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.updates = 0

    def _progress(self) -> float:
        return min(1.0, self.updates / self.cfg.decay_steps)

    def exploration_value(self) -> float:
        p = self._progress()
        if self.cfg.exploration == "epsilon":
            return self.cfg.epsilon_start + p * (self.cfg.epsilon_end-self.cfg.epsilon_start)
        if self.cfg.exploration == "softmax":
            return self.cfg.temperature_start
        if self.cfg.exploration == "dynamic_softmax":
            # Annealed Boltzmann exploration: broad coverage early, near-greedy late.
            return self.cfg.temperature_start * (
                self.cfg.temperature_end / self.cfg.temperature_start
            ) ** p
        raise ValueError(f"unknown exploration: {self.cfg.exploration}")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, device=self.device).unsqueeze(0)).squeeze(0)
        if not training:
            return int(q.argmax().item())
        value = self.exploration_value()
        if self.cfg.exploration == "epsilon":
            if self.rng.random() < value:
                return int(self.rng.integers(self.action_dim))
            return int(q.argmax().item())
        probabilities = torch.softmax(q / max(value, 1e-6), dim=0).cpu().numpy()
        return int(self.rng.choice(self.action_dim, p=probabilities))

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup):
            return None
        self.updates += 1
        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(ns, device=self.device)
        d = torch.as_tensor(d, device=self.device).unsqueeze(1)
        current = self.online(s).gather(1, a)
        with torch.no_grad():
            next_a = self.online(ns).argmax(1, keepdim=True)
            target = r + (1-d) * self.cfg.gamma * self.target(ns).gather(1, next_a)
        loss = nn.functional.smooth_l1_loss(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        if self.updates % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"model": self.online.state_dict(), "config": self.cfg.__dict__,
                    "updates": self.updates}, path)
