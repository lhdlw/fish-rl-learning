# V2 小预算实验归档

- config.json：顶层运行参数、平台、版本、运行开始时源码SHA256和本地Git状态。
- runs.csv：9组运行汇总；每个DQN运行15,000步，随机基线不训练。
- summary.csv：按3个种子汇总的均值和样本标准差，不是置信区间。
- runs/*/config.json：该运行的网络、奖励、种子、训练预算与时域参数。
- runs/*/training_episodes.csv：训练逐局记录，包含最后一个因总预算耗尽而未完成的局。
- runs/*/evaluation_episodes.csv：每组50局的评估环境种子和逐局指标，共450局。
- runs/*/summary.json：该运行的指标汇总、更新次数与耗时。
- runs/dqn_full__seed=0/model.pt：固定录像使用的模型，未依据评估择优。
- model_hashes.json：所有6个新模型的哈希；其余5个模型仅保留本地。

training_return是当组训练奖励的未折扣累计值，不是统一任务指标。
objective_return是5×吃鱼数−5×死亡次数；success_rate在逐局CSV中是0/1指示量。
time_limit表示局末到达时域上限；budget_cutoff表示训练总步数耗尽且该局尚未结束。
最后一局若有budget_cutoff=1，不能与完整训练局长度/总回报直接类比。
mean_loss是该局内更新损失的均值；epsilon是该段结束后的探索概率，衰减按更新次数推进。
随机基线虽保留统一配置字段，但不使用AgentConfig训练，没有训练日志或模型文件。

核心源文件路径在Windows运行记录中使用反斜杠，其他平台核对时需替换成相应路径分隔符。
本地提交标识可能与API同步后的远端提交标识不同，使用源码哈希核对实际代码。
耗时和图像文件不承诺跨设备逐字节一致；模型文件序列化哈希也不等于参数数值重现性测试。

方案：[PROTOCOL_V2.md](../../docs/PROTOCOL_V2.md)；解释：[EXPERIMENTS.md](../../docs/EXPERIMENTS.md)。
