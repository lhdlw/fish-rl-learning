"""
DQN训练脚本 - 大鱼吃小鱼
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from collections import deque
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fish_game import FishGame
from agent.dueling_dqn import DQNAgent


# ==================== 训练配置 ====================
NUM_EPISODES    = 1000      # 训练episode数
PRINT_EVERY     = 20        # 每N个episode打印一次
SAVE_EVERY      = 100       # 每N个episode保存checkpoint（用于训练痕迹视频）
RENDER_EVERY    = 0         # 每N个episode渲染一次(0=不渲染, 建议训练时不渲染)
MAX_STEPS       = 2000      # 单episode最大步数

MODEL_SAVE_DIR   = "models"
CHECKPOINT_DIR   = "models/checkpoints"   # 训练过程checkpoint独立目录
RESULT_SAVE_DIR  = "results"
MODEL_NAME       = "dueling_dqn_fish"

# 稠密/稀疏奖励开关
USE_DENSE_REWARD = True


def train():
    """主训练循环"""

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

    reward_type = "dense" if USE_DENSE_REWARD else "sparse"
    print(f"=" * 55)
    print(f"  大鱼吃小鱼 - Dueling DQN 训练")
    print(f"  奖励模式: {reward_type}")
    print(f"  训练集数: {NUM_EPISODES}")
    print(f"  设备: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
    print(f"=" * 55)

    # 初始化
    env = FishGame(render=False, dense_reward=USE_DENSE_REWARD)
    agent = DQNAgent()

    # 训练记录
    episode_rewards = []       # 每episode总奖励
    episode_scores = []        # 每episode吃鱼数
    episode_lengths = []       # 每episode步数
    reward_window = deque(maxlen=100)  # 最近100个episode的滑动窗口

    best_avg_reward = -float('inf')
    total_start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        episode_loss = 0.0
        update_count = 0

        for step in range(MAX_STEPS):
            # 选择动作
            action = agent.select_action(state, training=True)

            # 执行动作
            next_state, reward, done = env.step(action)

            # 存入经验池
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 更新网络
            loss = agent.update()
            if loss > 0:
                episode_loss += loss
                update_count += 1

            state = next_state
            total_reward += reward

            if done:
                break

        # 记录
        episode_rewards.append(total_reward)
        episode_scores.append(env.score)
        episode_lengths.append(env.total_steps)
        reward_window.append(total_reward)

        avg_reward = np.mean(reward_window) if len(reward_window) > 0 else total_reward
        avg_loss = episode_loss / max(update_count, 1)

        # 打印进度
        if episode % PRINT_EVERY == 0:
            elapsed = time.time() - total_start_time
            print(f"Ep {episode:>5d}/{NUM_EPISODES} | "
                  f"Reward: {total_reward:>7.2f} | "
                  f"Avg100: {avg_reward:>7.2f} | "
                  f"Score: {env.score:>3d} | "
                  f"Loss: {avg_loss:>6.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Elapsed: {elapsed:.0f}s")

        # 保存最佳模型
        if avg_reward > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward
            agent.save(os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_{reward_type}_best.pth"))

        # 定期保存checkpoint（用于训练痕迹视频）
        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_{reward_type}_ep{episode}.pth"))

        # 可选渲染
        if RENDER_EVERY > 0 and episode % RENDER_EVERY == 0:
            render_env = FishGame(render=True, dense_reward=USE_DENSE_REWARD)
            render_state = render_env.reset()
            render_done = False
            while not render_done:
                render_env.render()
                action = agent.select_action(render_state, training=False)
                render_state, _, render_done = render_env.step(action)
            render_env.close()

    # 保存最终模型
    agent.save(os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_{reward_type}_final.pth"))

    total_time = time.time() - total_start_time
    print(f"\n训练完成! 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")

    # 保存训练曲线数据
    np.savez(
        os.path.join(RESULT_SAVE_DIR, f"training_data_{reward_type}.npz"),
        rewards=episode_rewards,
        scores=episode_scores,
        lengths=episode_lengths,
    )

    # 绘制训练曲线
    plot_training_curves(episode_rewards, episode_scores, episode_lengths, reward_type)

    return episode_rewards, episode_scores


def plot_training_curves(rewards, scores, lengths, reward_type):
    """绘制训练曲线"""

    def smooth(data, window=50):
        """滑动平均平滑"""
        if len(data) < window:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        return smoothed

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"大鱼吃小鱼 - Dueling DQN 训练曲线 ({reward_type} reward)", fontsize=14)

    episodes = range(1, len(rewards) + 1)

    # 奖励曲线
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.3, color='steelblue', linewidth=0.5)
    ax.plot(episodes[len(episodes)-len(smooth(rewards)):], smooth(rewards),
            color='steelblue', linewidth=2, label='Smoothed (50)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 吃鱼数曲线
    ax = axes[1]
    ax.plot(episodes, scores, alpha=0.3, color='seagreen', linewidth=0.5)
    ax.plot(episodes[len(episodes)-len(smooth(scores)):], smooth(scores),
            color='seagreen', linewidth=2, label='Smoothed (50)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Fish Eaten')
    ax.set_title('Score per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 存活步数曲线
    ax = axes[2]
    ax.plot(episodes, lengths, alpha=0.3, color='coral', linewidth=0.5)
    ax.plot(episodes[len(episodes)-len(smooth(lengths)):], smooth(lengths),
            color='coral', linewidth=2, label='Smoothed (50)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Survival Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULT_SAVE_DIR, f"training_curves_{reward_type}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


if __name__ == "__main__":
    train()
