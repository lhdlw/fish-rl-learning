"""
使用训练好的模型进行游戏演示
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fish_game import FishGame
from agent.dueling_dqn import DQNAgent


def play(model_path: str, num_episodes: int = 5, dense_reward: bool = True):
    """
    加载模型并演示游戏

    Args:
        model_path: 模型文件路径
        num_episodes: 游玩episode数
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 train.py 或 experiment.py 进行训练")
        return

    env = FishGame(render=True, dense_reward=dense_reward)
    agent = DQNAgent()
    agent.load(model_path)

    print(f"\n{'='*50}")
    print(f"  大鱼吃小鱼 - 模型演示")
    print(f"  模型: {model_path}")
    print(f"  游玩 {num_episodes} 局")
    print(f"{'='*50}\n")

    total_rewards = []
    total_scores = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            env.render()

            # 处理退出事件
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
                    if event.key == pygame.K_SPACE:
                        # 暂停/继续
                        paused = True
                        while paused:
                            for e in pygame.event.get():
                                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                    paused = False
                                if e.type == pygame.QUIT:
                                    env.close()
                                    return

            action = agent.select_action(state, training=False)
            state, reward, done = env.step(action)
            total_reward += reward
            step += 1

        total_rewards.append(total_reward)
        total_scores.append(env.score)
        print(f"Episode {ep}: Reward={total_reward:.2f}, Score={env.score}, Steps={step}")

    env.close()

    print(f"\n{'='*50}")
    print(f"  演示完成!")
    print(f"  平均奖励: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"  平均吃鱼: {sum(total_scores)/len(total_scores):.1f}")
    print(f"  总吃鱼数: {sum(total_scores)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大鱼吃小鱼 - 模型演示")
    parser.add_argument("--model", type=str, default=os.path.join(os.path.dirname(__file__), "models", "dueling_dqn_dense_best.pth"),
                        help="模型文件路径")
    parser.add_argument("--episodes", type=int, default=5,
                        help="游玩episode数")
    parser.add_argument("--sparse", action="store_true",
                        help="使用稀疏奖励环境")
    args = parser.parse_args()

    play(args.model, args.episodes, dense_reward=not args.sparse)
