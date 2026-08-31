"""
Dueling DQN 网络结构和智能体
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple


# ==================== Dueling DQN 网络 ====================

class DuelingDQN(nn.Module):
    """
    Dueling DQN网络结构

    共享层: Linear(21→128) → ReLU → Linear(128→64) → ReLU
    价值分支: Linear(64→1)
    优势分支: Linear(64→4)

    Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
    """

    def __init__(self, state_dim: int = 21, action_dim: int = 4):
        super().__init__()

        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 价值分支 V(s)
        self.value = nn.Linear(64, 1)

        # 优势分支 A(s, a)
        self.advantage = nn.Linear(64, action_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: shape (batch, state_dim)

        Returns:
            Q值: shape (batch, action_dim)
        """
        features = self.feature(state)

        value = self.value(features)          # (batch, 1)
        advantage = self.advantage(features)  # (batch, action_dim)

        # Q = V + A - mean(A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q


# ==================== 经验回放池 ====================

class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity: int = 80000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)

        states      = np.array([t[0] for t in batch], dtype=np.float32)
        actions     = np.array([t[1] for t in batch], dtype=np.int64)
        rewards     = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones       = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ==================== DQN智能体 ====================

class DQNAgent:
    """
    Dueling DQN智能体

    超参数:
        lr=1e-4, gamma=0.93
        经验回放池 80000, 目标网络每300步更新
        ε-greedy: 初始0.9 → 最终0.02
    """

    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.93,
        buffer_capacity: int = 80000,
        target_update_freq: int = 300,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 50000,
        batch_size: int = 64,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 网络
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # 训练步数计数器
        self.train_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-greedy策略）

        Args:
            state: 状态向量 (state_dim,)
            training: 是否训练模式

        Returns:
            action: 0=上, 1=下, 2=左, 3=右
        """
        if training:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-self.train_steps / self.epsilon_decay)
        else:
            epsilon = 0.0  # 评估模式：纯贪心

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        # 贪心选择
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self) -> float:
        """
        从经验回放池采样并更新网络

        Returns:
            loss: 本次更新的损失值
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.train_steps += 1

        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前Q值
        current_q = self.policy_net(states).gather(1, actions)

        # 目标Q值（Double DQN: 用policy_net选动作，target_net评估）
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 损失 & 优化
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新目标网络
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
        }, path)
        print(f"模型已保存至: {path}")

    def load(self, path: str):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.train_steps = ckpt['train_steps']
        print(f"模型已加载: {path} (训练步数: {self.train_steps})")

    @property
    def epsilon(self) -> float:
        """当前ε值"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.train_steps / self.epsilon_decay)
