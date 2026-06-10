# Mario 强化学习内在奖励实验分析报告

## 1. 摘要

本实验在 `SuperMarioBros-1-1-v0` 环境中固定 PPO 作为基础强化学习框架，对比三种内在奖励方法：PPO + NGU、PPO + RE3、PPO + E3B。三种方法共享同一环境封装、神经网络结构、PPO 训练流程和主要超参数，差异集中在内在奖励模块。

三组最终实验均完成 1,000,000 timestep 训练，训练设备为 CUDA，动作空间为 `SIMPLE_MOVEMENT`，动作数为 7，输入观测为 4 帧堆叠的 84×84 灰度图像。训练阶段使用外部奖励与内在奖励的组合进行 PPO 更新；评估和视频录制阶段不计算内在奖励，因此报告中的评估 `total_reward` 等同于环境外部奖励。

主要观察结果如下：

- 在 `SuperMarioBros-1-1-v0` 的训练期 validation sample 指标中，PPO + NGU 的 best mean total reward 最高，为 2334.0；PPO + E3B 为 2016.5；PPO + RE3 为 1860.5。
- 在 `SuperMarioBros-1-1-v0` 的 1000 episode sample 评估中，PPO + NGU 和 PPO + RE3 都达到 `max_x_pos=3161`；PPO + E3B 的最高 `max_x_pos` 为 2475。
- 在 `SuperMarioBros-1-2-v0` 的 transfer 评估中，PPO + NGU 找到最高单局迁移轨迹，best reward 为 2070，best `max_x_pos` 为 2272；但从 1000 episode 的 mean / median 指标看，PPO + E3B 和 PPO + RE3 的平均表现略高。
- 当前实验只包含 seed 1，不能据此声明统计显著性；结论应限定为“在本次实验设置和样本下观察到”。

## 2. 实验目标

实验目标是比较三种 PPO 内在奖励方法在 Mario 环境中的探索与任务表现。具体做法是固定 PPO backbone，只替换内在奖励模块，从而减少因基础 RL 算法不同造成的混杂影响。

本报告覆盖以下三种方法：

| 方法 | 训练入口 | 配置文件 | 内在奖励类型 |
|---|---|---|---|
| PPO + NGU | `experiments/ppo_ngu.py` | `configs/ppo_ngu.yaml` | `ngu` |
| PPO + RE3 | `experiments/ppo_re3.py` | `configs/ppo_re3.yaml` | `re3` |
| PPO + E3B | `experiments/ppo_e3b.py` | `configs/ppo_e3b.yaml` | `e3b` |

本报告不覆盖 DQN、DDPG、SAC 或 SAC-Discrete。实验包中的运行报告明确说明第一轮主线固定 PPO backbone，未运行这些算法。

## 3. 代码与实现依据

核心实现文件如下：

| 模块 | 文件路径 | 作用 |
|---|---|---|
| PPO 主训练逻辑 | `01_project_source/mario_rl/algorithms/ppo.py` | 构建 Actor-Critic、收集 rollout、计算 PPO loss、保存 checkpoint、执行 validation/test |
| 内在奖励适配 | `01_project_source/mario_rl/rewards/rllte_intrinsic.py` | 将 `ngu`、`re3`、`e3b` 等配置项映射到 RLLTE reward module |
| 环境创建 | `01_project_source/mario_rl/envs/make_env.py` | 构建 Super Mario 环境，应用动作空间、跳帧、观测处理和 episode 长度限制 |
| 环境 wrapper | `01_project_source/mario_rl/envs/wrappers.py` | 实现 frame skip、灰度缩放、frame stack、最大步数截断 |
| 三种方法入口 | `01_project_source/experiments/*.py` | 分别加载对应 YAML 配置并调用 PPO `main` |
| 长训练脚本 | `01_project_source/remote_run_ppo_intrinsic_1m.sh` | 对三种方法执行 1,000,000 timestep 训练，并录制 checkpoint 视频 |
| checkpoint 评估 | `01_project_source/mario_rl/evaluate_checkpoint.py` | 从 checkpoint 采样或 greedy 执行动作，记录 reward、x 位置和视频 |
| 双最佳视频评估 | `01_project_source/tools/record_dual_best_checkpoint_videos.py` | 在同一批 sampled episodes 中分别保存 reward-best 与 x-best 视频 |

PPO 网络为卷积 Actor-Critic。输入经三层卷积和全连接层编码后，actor 输出离散动作 logits，critic 输出状态价值。动作分布使用 categorical distribution；sample mode 从分布采样，greedy mode 取最大 logit 动作。

内在奖励模块通过 `build_intrinsic_module` 创建。PPO 更新时先计算 rollout 外部奖励和内在奖励，再形成 `rollout_total_rewards = rollout_ext_rewards + rollout_int_rewards_tensor`，并在配置开启时将总奖励裁剪到 `[-1, 1]`。评估函数中 `int_reward` 固定为 0，因此 validation、test 和视频评估中的 `total_reward` 是外部环境奖励。

## 4. 实验设置

### 4.1 环境设置

| 项目 | 设置 |
|---|---|
| 训练环境 | `SuperMarioBros-1-1-v0` |
| 主要同关卡评估环境 | `SuperMarioBros-1-1-v0` |
| 迁移评估环境 | `SuperMarioBros-1-2-v0` |
| 动作空间 | `SIMPLE_MOVEMENT` |
| 动作数 | 7 |
| frame skip | 4 |
| frame stack | 4 |
| 图像尺寸 | 84×84 |
| 单 episode 最大 decision steps | 3000 |
| 观测处理 | RGB 转灰度，缩放到 84×84，堆叠最近 4 帧 |

### 4.2 PPO 训练设置

三种方法共享以下 PPO 设置：

| 项目 | 设置 |
|---|---:|
| total timesteps | 1,000,000 |
| rollout steps | 128 |
| update epochs | 2 |
| minibatch size | 64 |
| learning rate | 0.0001 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clip coefficient | 0.1 |
| value loss coefficient | 0.5 |
| entropy coefficient | 0.05 |
| max grad norm | 0.5 |
| target KL | 0.03 |
| reward clipping | true |
| reward clip range | [-1.0, 1.0] |
| eval interval | 1000 |
| eval episodes | 2 |
| test episodes | 2 |
| eval modes | greedy, sample |
| best checkpoint selection | sample validation mean total reward |
| seed | 1 |
| device | cuda |

源配置文件中的默认 `total_timesteps` 为 5000；实际 1M 训练由长训练脚本和最终 run 的 `config.yaml` 覆盖为 1,000,000。最终 run 的 `summary.json` 均确认 `global_step=1000000`。

### 4.3 内在奖励设置

| 方法 | beta | latent dim | 其他关键参数 |
|---|---:|---:|---|
| PPO + NGU | 0.001 | 32 | `k=10`，`kernel_cluster_distance=0.008`，`sm=8.0`，`mrs=5.0` |
| PPO + RE3 | 0.001 | 128 | `storage_size=1000`，`k=5`，`average_entropy=false` |
| PPO + E3B | 0.001 | 128 | `ridge=0.1`，`batch_size=32`，`update_proportion=1.0` |

三种方法均关闭 reward normalization 和 observation normalization：`rwd_norm_type=none`，`obs_norm_type=none`。

### 4.4 实现修复

实验过程中对本地 RLLTE 代码进行了两处离散动作兼容修复：

| 文件 | 修复内容 |
|---|---|
| `rllte/rllte/xplore/reward/pseudo_counts.py` | 修复离散动作下 `CrossEntropyLoss(reduction="none")` 返回 1D loss 时 mask expand 维度不匹配问题 |
| `rllte/rllte/xplore/reward/e3b.py` | 对 E3B 的离散动作 loss mask 做相同维度兼容处理 |

该修复用于保证离散动作环境下内在奖励模块能够运行，不改变 PPO 主训练流程。

## 5. 训练完成情况

三组最终 1M run 均完成训练，并生成 summary、训练指标、更新指标、validation/test 指标、checkpoint 和视频。

| 方法 | 最终 run 目录 | global step | 训练 episodes | best validation mean reward | best step |
|---|---|---:|---:|---:|---:|
| PPO + NGU | `runs/ppo_ngu_1m_seed1_1780568836` | 1,000,000 | 7547 | 2334.0 | 738000 |
| PPO + RE3 | `runs/ppo_re3_1m_seed1_1780587129` | 1,000,000 | 7669 | 1860.5 | 735000 |
| PPO + E3B | `runs/ppo_e3b_1m_seed1_1780604303` | 1,000,000 | 7582 | 2016.5 | 405000 |

按 validation sample mean total reward 排序：

```text
PPO + NGU > PPO + E3B > PPO + RE3
```

该排序只适用于当前 seed、当前 validation 设置和 sample evaluation。由于每次 validation 只使用 2 episodes，validation best 具有较高采样波动，不能单独作为最终泛化结论。

训练末尾的 update metrics 显示三种方法均存在非零 rollout-level intrinsic reward 信号：

| 方法 | last rollout intrinsic reward mean | last rollout intrinsic reward sum | entropy | approx KL | dominant action frac |
|---|---:|---:|---:|---:|---:|
| PPO + NGU | 0.037326 | 2.388838 | 1.540470 | 0.000155 | 0.406250 |
| PPO + RE3 | 0.000297 | 0.019014 | 1.565346 | 0.000178 | 0.421875 |
| PPO + E3B | 0.000551 | 0.035277 | 1.644134 | 0.000231 | 0.390625 |

NGU 在训练末尾的 intrinsic reward 数值明显高于 RE3 和 E3B。但该现象只能说明该实现和该阶段下 NGU 输出的内在奖励尺度更大，不能直接证明它是最终表现差异的唯一原因。

## 6. 同关卡评估结果：`SuperMarioBros-1-1-v0`

最终同关卡评估使用 1000 episode sample rollout。每种方法在同一批 sampled episodes 中分别记录 reward-best 和 x-best 轨迹。

### 6.1 最佳单局结果

| 方法 | reward-best episode | best reward | reward-best max x | x-best episode | best max x | x-best reward |
|---|---:|---:|---:|---:|---:|---:|
| PPO + NGU | 568 | 3014 | 3161 | 294 | 3161 | 3009 |
| PPO + RE3 | 21 | 3017 | 3161 | 21 | 3161 | 3017 |
| PPO + E3B | 567 | 2344 | 2472 | 451 | 2475 | 2343 |

PPO + NGU 和 PPO + RE3 都达到 `max_x_pos=3161`。PPO + RE3 的 best reward 略高于 PPO + NGU，差值为 3。PPO + E3B 在该批评估中未达到 3161，最高 `max_x_pos` 为 2475。

### 6.2 1000 episode 分布统计

| 方法 | mean reward | median reward | max reward | mean x | median x | max x | x≥3000 次数 | x≥2000 次数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO + NGU | 724.680 | 631.0 | 3014 | 808.416 | 707.0 | 3161 | 2 | 13 |
| PPO + RE3 | 703.120 | 626.5 | 3017 | 788.452 | 704.0 | 3161 | 1 | 9 |
| PPO + E3B | 713.261 | 626.5 | 2344 | 793.281 | 703.0 | 2475 | 0 | 4 |

同关卡评估的主要结论是：

- 若以 best reward 为指标，PPO + RE3 最高，PPO + NGU 极接近，PPO + E3B 低于前两者。
- 若以 mean reward、mean x 或达到高 x 区间的次数为指标，PPO + NGU 略优于 PPO + RE3 和 PPO + E3B。
- PPO + E3B 的均值不差，但缺少达到 `x=3161` 的高表现轨迹。

因此，同关卡评估不能简单写成某一个方法在所有指标上最好。更准确的表述是：PPO + NGU 的整体采样分布略优，PPO + RE3 出现了最高 reward 的单局轨迹，二者都能在当前批次中达到 1-1 的最高记录位置；PPO + E3B 表现相对稳定但最高轨迹不及 NGU 和 RE3。

## 7. 迁移评估结果：`SuperMarioBros-1-2-v0`

迁移评估使用在 `SuperMarioBros-1-1-v0` 训练得到的 best checkpoint，直接在 `SuperMarioBros-1-2-v0` 上进行 1000 episode sample rollout。该评估用于观察从 1-1 到 1-2 的跨关卡迁移表现。

### 7.1 最佳单局结果

| 方法 | reward-best episode | best reward | reward-best max x | x-best episode | best max x | x-best reward |
|---|---:|---:|---:|---:|---:|---:|
| PPO + NGU | 971 | 2070 | 2272 | 971 | 2272 | 2070 |
| PPO + RE3 | 261 | 1747 | 1940 | 261 | 1940 | 1747 |
| PPO + E3B | 617 | 1810 | 1955 | 617 | 1955 | 1810 |

PPO + NGU 在 1-2 transfer 中获得最高单局 reward 和最高 `max_x_pos`。PPO + E3B 的 best reward 和 best x 均略高于 PPO + RE3。

### 7.2 1000 episode 分布统计

| 方法 | mean reward | median reward | max reward | mean x | median x | max x | x≥2000 次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO + NGU | 450.942 | 531.0 | 2070 | 543.260 | 649.0 | 2272 | 1 |
| PPO + RE3 | 463.436 | 533.0 | 1747 | 561.612 | 654.5 | 1940 | 0 |
| PPO + E3B | 477.229 | 564.0 | 1810 | 563.894 | 661.0 | 1955 | 0 |

迁移评估的主要结论是：

- PPO + NGU 找到了最强的单局迁移轨迹，是三种方法中唯一达到 `x≥2000` 的方法。
- 从 mean reward、median reward、mean x 和 median x 看，PPO + E3B 略高于另外两种方法。
- PPO + RE3 的平均表现介于 NGU 和 E3B 之间，但没有出现超过 `x=2000` 的轨迹。

因此，1-2 transfer 结果应分成两个层面报告：PPO + NGU 的 best observed trajectory 最强；PPO + E3B 的 1000 episode 平均/中位数表现略好。

## 8. 方法对比分析

### 8.1 PPO + NGU

PPO + NGU 在 validation sample 指标中获得最高 best mean reward，并在 1-1 与 1-2 的评估中都出现了较强单局轨迹。1-1 中，NGU 达到 `max_x_pos=3161`；1-2 transfer 中，NGU 达到 `max_x_pos=2272`，是三种方法中最高。

NGU 的特点是 best trajectory 强，尤其在迁移环境中优势明显。但 1-2 的 mean / median 指标并非最高，说明其高表现轨迹不是大多数 episodes 的典型表现，而是 sampled rollout 中出现的较强个例。

### 8.2 PPO + RE3

PPO + RE3 在 1-1 的 1000 episode 评估中获得最高 best reward：3017，并达到 `max_x_pos=3161`。这说明 RE3 checkpoint 能在同关卡采样中产生完成级别的高表现轨迹。

RE3 的 validation best mean reward 低于 NGU 和 E3B；1-1 的 mean reward 与 mean x 也低于 NGU 和 E3B。该结果表示 RE3 在当前批次下具备高峰值表现，但整体采样分布不占优。

### 8.3 PPO + E3B

PPO + E3B 的 validation best mean reward 排名第二。在 1-1 评估中，E3B 的 mean reward 高于 RE3、低于 NGU，但 best x 明显低于 NGU 和 RE3，未达到 `x=3161`。

在 1-2 transfer 中，E3B 的 mean reward、median reward、mean x 和 median x 均为三种方法中最高，但 best trajectory 不及 NGU。这说明 E3B 在迁移评估中的平均表现较稳，但没有找到 NGU 那样远的单局轨迹。

### 8.4 sample mode 与 greedy mode

最终 checkpoint 选择使用 sample validation mean total reward。best summary 中的 greedy 结果整体弱于 sample 结果，尤其 PPO + NGU 的 greedy validation/test 在 best step 上停留在很低的 x 位置。该现象说明本实验中的策略质量主要通过随机采样表现出来，而不是通过 greedy action 表现出来。

因此，本报告的主要结果以 sample evaluation 为准。若后续需要比较部署时的确定性策略，应单独设计 greedy evaluation 实验，而不能直接用当前 sample 结果替代。

## 9. 局限性

本实验存在以下限制：

1. **只有单个 seed。** 当前三种方法均使用 seed 1，不能判断结果是否具有统计显著性。
2. **validation episodes 较少。** 训练中的 validation 每次仅 2 episodes，best validation 指标容易受采样波动影响。
3. **最终评估依赖 sample rollout。** 1000 episode 评估能更好观察随机策略的高表现轨迹，但不能代表 greedy 部署表现。
4. **没有 extrinsic-only PPO 对照。** 当前报告比较三种内在奖励方法，不能直接量化“加入内在奖励相对纯 PPO 的提升”。
5. **没有消融实验。** 不能证明某个具体超参数、RLLTE 模块内部机制或 intrinsic reward 尺度单独导致结果差异。
6. **transfer 只测试 1-2。** 当前迁移结论只适用于从 1-1 checkpoint 到 1-2 环境的测试，不能泛化到所有 Mario 关卡。

## 10. 结论

在本次 Mario 强化学习实验中，三种 PPO 内在奖励方法均完成 1,000,000 timestep 训练，并产生可复核的训练指标、validation/test 指标和 1000 episode 评估视频。固定 PPO backbone 后，三种方法的主要差异来自内在奖励模块。

综合当前证据，可以得到以下结论：

1. **PPO + NGU 在 validation sample 指标和 transfer best trajectory 上最突出。** 它在 1-1 validation 中获得最高 best mean reward，并在 1-2 transfer 中达到最高 `max_x_pos=2272`。
2. **PPO + RE3 在 1-1 中产生最高 reward 单局轨迹。** 它在 1000 episode 同关卡评估中获得 best reward 3017，并达到 `max_x_pos=3161`。
3. **PPO + E3B 在 1-2 transfer 的均值和中位数指标上表现较稳。** 它没有达到 NGU 的最佳迁移距离，但在 mean/median reward 和 x 位置上略高。
4. **不同评价指标会改变方法排序。** 若看 validation best，排序为 NGU > E3B > RE3；若看 1-1 best reward，RE3 略高于 NGU；若看 1-2 best trajectory，NGU 最强；若看 1-2 mean/median，E3B 略优。

因此，最稳妥的总体表述是：在本次 seed 1、1M timestep、sample evaluation 设置下，NGU 展现出最强的探索峰值和迁移峰值，RE3 在同关卡最高单局 reward 上略优，E3B 在迁移分布均值上更稳定。由于缺少多 seed 与消融实验，当前结果应视为实验观察，而不是统计证明。

## 11. 证据文件

本报告依据以下项目文件和实验产物整理：

| 证据类型 | 文件路径 |
|---|---|
| 源码 | `01_project_source/mario_rl/algorithms/ppo.py` |
| 内在奖励适配 | `01_project_source/mario_rl/rewards/rllte_intrinsic.py` |
| 环境封装 | `01_project_source/mario_rl/envs/make_env.py`，`01_project_source/mario_rl/envs/wrappers.py` |
| 方法配置 | `01_project_source/configs/ppo_ngu.yaml`，`01_project_source/configs/ppo_re3.yaml`，`01_project_source/configs/ppo_e3b.yaml` |
| 方法入口 | `01_project_source/experiments/ppo_ngu.py`，`01_project_source/experiments/ppo_re3.py`，`01_project_source/experiments/ppo_e3b.py` |
| 训练脚本 | `01_project_source/remote_run_ppo_intrinsic_1m.sh` |
| 训练运行报告 | `02_local_outputs/RUN_REPORT_ppo_ngu_re3_e3b.md` |
| 迁移评估报告 | `02_local_outputs/RUN_REPORT_transfer_1_2_1000ep.md` |
| 同批次双最佳评估报告 | `02_local_outputs/RUN_REPORT_dual_best_1_1_1_2_1000ep.md` |
| 完整训练 run | `03_remote_training_artifacts/.../runs/ppo_ngu_1m_seed1_1780568836` |
| 完整训练 run | `03_remote_training_artifacts/.../runs/ppo_re3_1m_seed1_1780587129` |
| 完整训练 run | `03_remote_training_artifacts/.../runs/ppo_e3b_1m_seed1_1780604303` |
| 1000 episode 日志 | `03_remote_training_artifacts/.../long_logs/*dual*1000ep.log` |
| 视频结果 | `02_local_outputs/*.mp4` |
