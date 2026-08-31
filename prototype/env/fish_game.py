"""
大鱼吃小鱼 - Pygame游戏环境
状态使用向量数据，适合强化学习训练
"""

import pygame
import random
import numpy as np
from typing import Tuple, List

# ==================== 游戏参数 ====================
WIDTH   = 800
HEIGHT  = 600
FPS     = 30

FISH_SMALL_SIZE  = 25      # 小鱼尺寸
FISH_BIG_SIZE    = 55      # 大鱼尺寸
SELF_START_SIZE  = 30      # 我方初始尺寸
SELF_SPEED       = 6       # 我方移动速度
MAX_NPC          = 6       # 状态向量最多保存的NPC鱼数量
MIN_NPC_COUNT    = 4       # 屏幕最少维持的NPC数量

NPC_SPAWN_Y_MIN  = 50
NPC_SPAWN_Y_MAX  = HEIGHT - 50
NPC_SPEED_MIN    = 1.5
NPC_SPEED_MAX    = 3.0
BIG_FISH_PROB    = 0.4     # 生成大鱼的概率

CLIP_BOUNDARY    = 20      # 我方鱼边界限制
COLLISION_SCALE  = 2.2     # 碰撞判定缩放
IDLE_PENALTY     = 0.1     # 静止/顶墙惩罚（检测位置是否真的变化）


class FishGame:
    """
    大鱼吃小鱼游戏环境

    状态空间: [self_x, self_y, self_size, fish1_x, fish1_y, fish1_size, ...]
    共 3 + MAX_NPC*3 = 21 维，不足补0

    动作空间: 0=上, 1=下, 2=左, 3=右 (4个离散动作)
    """

    def __init__(self, render: bool = False, dense_reward: bool = True,
                 render_mode: str = None):
        """
        Args:
            render: 是否渲染画面（兼容旧接口，等价于 render_mode="human"）
            dense_reward: True=稠密塑形奖励, False=仅稀疏奖励
            render_mode: None | "human" | "rgb_array"
                None=不渲染, "human"=窗口渲染, "rgb_array"=无头渲染返回帧数组
        """
        # render_mode 优先；未指定时用 render 兼容
        if render_mode is None:
            render_mode = "human" if render else None
        self.render_mode = render_mode
        self.dense_reward = dense_reward

        # 渲染相关
        self.screen = None      # human模式的窗口surface
        self.canvas = None      # 实际绘制目标surface（human=screen, rgb_array=离屏）
        self.clock = None
        self.font = None

        # 游戏状态
        self.self_x = WIDTH // 2
        self.self_y = HEIGHT // 2
        self.self_size = SELF_START_SIZE
        self.self_speed = SELF_SPEED
        self.npc_fish: List[List[float]] = []
        self.score = 0              # 吃鱼计数
        self.total_steps = 0        # 总步数

        # 初始化pygame（仅渲染模式）
        if self.render_mode is not None:
            self._init_render()

    def _init_render(self):
        """初始化pygame渲染"""
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("大鱼吃小鱼 - DQN训练")
            self.canvas = self.screen
            self.clock = pygame.time.Clock()
        else:  # rgb_array：离屏surface，无窗口
            self.screen = None
            self.canvas = pygame.Surface((WIDTH, HEIGHT))
            self.clock = None
        self.font = pygame.font.Font(None, 24)

    def reset(self) -> np.ndarray:
        """重置游戏，返回初始状态向量"""
        self.self_x = WIDTH // 2
        self.self_y = HEIGHT // 2
        self.self_size = SELF_START_SIZE
        self.score = 0
        self.total_steps = 0
        self.npc_fish.clear()

        # 生成初始NPC鱼
        for _ in range(3):
            self._spawn_one_fish()

        return self._get_state_vector()

    def _spawn_one_fish(self):
        """从屏幕左侧或右侧随机生成一条NPC鱼"""
        side = random.choice(["left", "right"])
        is_big = random.random() < BIG_FISH_PROB

        sz = FISH_BIG_SIZE if is_big else FISH_SMALL_SIZE
        y_pos = random.randint(NPC_SPAWN_Y_MIN, NPC_SPAWN_Y_MAX)

        if side == "left":
            x_pos = 0.0
            direction = 1
        else:
            x_pos = float(WIDTH)
            direction = -1

        speed = random.uniform(NPC_SPEED_MIN, NPC_SPEED_MAX)
        self.npc_fish.append([x_pos, y_pos, sz, speed, direction])

    def _get_state_vector(self) -> np.ndarray:
        """构建固定长度的状态向量"""
        vec = [self.self_x / WIDTH,
               self.self_y / HEIGHT,
               self.self_size / 100.0]

        for i in range(MAX_NPC):
            if i < len(self.npc_fish):
                x, y, sz, _, _ = self.npc_fish[i]
                vec.append(x / WIDTH)
                vec.append(y / HEIGHT)
                vec.append(sz / 100.0)
            else:
                vec.extend([0.0, 0.0, 0.0])

        return np.array(vec, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        执行一步动作

        Args:
            action: 0=上, 1=下, 2=左, 3=右

        Returns:
            next_state: 下一状态向量
            reward: 即时奖励
            done: 是否终止
        """
        reward = 0.0
        done = False
        self.total_steps += 1

        # ===== 1. 我方鱼移动 =====
        prev_x, prev_y = self.self_x, self.self_y   # 记录移动前位置，用于静止检测

        if action == 0:      # 上
            self.self_y -= self.self_speed
        elif action == 1:    # 下
            self.self_y += self.self_speed
        elif action == 2:    # 左
            self.self_x -= self.self_speed
        elif action == 3:    # 右
            self.self_x += self.self_speed

        # 边界约束
        self.self_x = np.clip(self.self_x, CLIP_BOUNDARY, WIDTH - CLIP_BOUNDARY)
        self.self_y = np.clip(self.self_y, CLIP_BOUNDARY, HEIGHT - CLIP_BOUNDARY)

        # ===== 2. NPC鱼移动 & 边界移除 =====
        for fish in self.npc_fish:
            fish[0] += fish[3] * fish[4]  # x += speed * direction

        self.npc_fish = [f for f in self.npc_fish if -50 < f[0] < WIDTH + 50]

        # ===== 3. 补充NPC鱼 =====
        while len(self.npc_fish) < MIN_NPC_COUNT:
            self._spawn_one_fish()

        # ===== 4. 碰撞检测 =====
        eaten_indices = []
        for idx, f in enumerate(self.npc_fish):
            fx, fy, fsz = f[0], f[1], f[2]
            dist = np.hypot(self.self_x - fx, self.self_y - fy)
            collision_threshold = (self.self_size + fsz) / COLLISION_SCALE

            if dist < collision_threshold:
                if fsz < self.self_size:
                    # 吃掉小鱼
                    reward += 5.0
                    self.self_size += 2
                    self.score += 1
                    eaten_indices.append(idx)
                else:
                    # 撞上大鱼
                    reward += -5.0
                    done = True
                    break

        # 移除被吃掉的鱼
        for idx in sorted(eaten_indices, reverse=True):
            if idx < len(self.npc_fish):
                del self.npc_fish[idx]

        # ===== 5. 塑形奖励（仅稠密模式） =====
        if not done and self.dense_reward:
            min_dist_small = 9999.0
            min_dist_big   = 9999.0

            for f in self.npc_fish:
                fx, fy, fsz = f[0], f[1], f[2]
                d = np.hypot(self.self_x - fx, self.self_y - fy)
                if fsz < self.self_size:
                    if d < min_dist_small:
                        min_dist_small = d
                else:
                    if d < min_dist_big:
                        min_dist_big = d

            # 靠近小鱼奖励（距离越近奖励越大）
            reward += 0.15 * np.exp(-min_dist_small / 120.0)
            # 靠近大鱼惩罚：仅当大鱼进入 130px 内才罚，距离越近惩罚越强（主动躲避）
            if min_dist_big < 130:
                reward -= 0.5 * np.exp(-min_dist_big / 60.0)
            # 静止/顶墙惩罚：位置没变化才罚（避免agent躺着赚塑形奖励）
            moved = (self.self_x != prev_x) or (self.self_y != prev_y)
            if not moved:
                reward -= IDLE_PENALTY

        # ===== 6. 防止无限循环 =====
        if self.total_steps > 2000:
            done = True

        next_state = self._get_state_vector()
        return next_state, reward, done

    def render(self):
        """
        渲染当前游戏画面

        Returns:
            human模式: None（画面直接显示在窗口）
            rgb_array模式: numpy数组 (H, W, 3) RGB
            None模式: None
        """
        if self.render_mode is None:
            return None

        # 处理事件，防止窗口卡死（仅human模式）
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

        # 背景（海洋渐变）
        self.canvas.fill((10, 50, 120))
        # 简单装饰
        pygame.draw.rect(self.canvas, (15, 70, 140), (0, 0, WIDTH, 30))  # 顶部深色条
        pygame.draw.rect(self.canvas, (15, 70, 140), (0, HEIGHT-30, WIDTH, 30))  # 底部深色条

        # 绘制NPC鱼
        for f in self.npc_fish:
            fx, fy, fsz = int(f[0]), int(f[1]), int(f[2])
            if fsz > self.self_size:
                # 大鱼：红色
                color = (220, 50, 50)
            else:
                # 小鱼：绿色
                color = (50, 200, 100)

            # 鱼身（椭圆）
            pygame.draw.ellipse(self.canvas, color,
                              (fx - fsz, fy - fsz // 2, fsz * 2, fsz))
            # 眼睛
            eye_x = fx + fsz // 2 if f[4] > 0 else fx - fsz // 2
            pygame.draw.circle(self.canvas, (255, 255, 255), (eye_x, fy - fsz // 4), fsz // 5)
            pygame.draw.circle(self.canvas, (0, 0, 0), (eye_x, fy - fsz // 4), fsz // 10)

        # 绘制我方鱼（金鱼色）
        sx, sy, ssz = int(self.self_x), int(self.self_y), int(self.self_size)
        pygame.draw.ellipse(self.canvas, (255, 200, 50),
                          (sx - ssz, sy - ssz // 2, ssz * 2, ssz))
        # 眼睛
        pygame.draw.circle(self.canvas, (255, 255, 255), (sx + ssz // 2, sy - ssz // 4), ssz // 4)
        pygame.draw.circle(self.canvas, (0, 0, 0), (sx + ssz // 2, sy - ssz // 4), ssz // 8)

        # HUD信息
        score_text = self.font.render(f"Score: {self.score}  Size: {self.self_size}",
                                       True, (255, 255, 255))
        self.canvas.blit(score_text, (10, 5))

        # 按模式输出
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        else:  # rgb_array：返回帧数组 (H, W, 3) RGB
            frame = pygame.surfarray.array3d(self.canvas)  # (W, H, 3)
            frame = frame.transpose(1, 0, 2)               # (H, W, 3)
            return np.ascontiguousarray(frame)

    def close(self):
        """关闭渲染窗口"""
        if self.render_mode is not None:
            pygame.quit()
