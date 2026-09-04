# 原版小游戏（历史学习原型）

此目录保留21维观测、四方向动作的Pygame环境，及Dueling/Double DQN智能体。
从仓库根目录运行以下命令观看历史原型。训练请进入本目录后运行 `python train.py`。

```bash
python -m pip install -r prototype/requirements.txt
python prototype/play.py --model prototype/models/dueling_dqn_dense_best.pth --episodes 3
```

![历史演示，版本未完整绑定](../media/ai_play.gif)


## 已知局限

- 状态未包含NPC速度和方向，单帧观测不是完整马尔可夫状态。
- 原训练没有显式设置种子；2000步训练上限与环境的2001步截断存在差异。
- 训练奖励、代码版本、历史模型没有完整绑定，不能严格追溯演示模型的配置。
- `dueling_dqn_dense_best.pth` 是原文件名；旧对照脚本会把末次模型保存到该名称。
- 存在训练和评估、停止规则及界面事件处理等需要改进之处。

保留此目录是为了展示学习起点，不把它当作严谨对照实验。
旧报告因结论与数据/配置不同步没有纳入仓库。新实验请使用根目录的实验脚本。
模型只使用仓库中自有历史权重，加载设置为 `weights_only=True`；不要随意加载不可信权重。
