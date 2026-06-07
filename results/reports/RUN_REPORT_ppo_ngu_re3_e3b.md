# PPO + NGU / RE3 / E3B 运行报告

- 生成时间：2026-06-05T09:05:56
- 本次只运行：PPO + NGU、PPO + RE3、PPO + E3B
- 未运行：DQN / DDPG / SAC / SAC-Discrete
- 原因：第一轮主线固定 PPO backbone，避免 RL backbone 差异成为混杂变量。
- 说明：当前训练是外部奖励 + 内在奖励的 PPO 训练链路；本报告只说明代码入口、训练链路、日志、checkpoint 和视频产出情况，不说明方法性能优劣。

## 静态检查结果

| 检查项 | NGU | RE3 | E3B | 证据 |
| --- | --- | --- | --- | --- |
| 入口脚本存在 | 通过 | 通过 | 通过 | experiments/*.py |
| 配置文件存在 | 通过 | 通过 | 通过 | configs/*.yaml |
| reward type 正确 | 通过 | 通过 | 通过 | YAML reward.intrinsic.type |
| reward module 映射存在 | 通过 | 通过 | 通过 | mario_rl/rewards/rllte_intrinsic.py |

## Smoke test 结果

| 方法 | 命令 | 退出码 | run_dir | 结果 | 失败说明 |
| --- | --- | ---: | --- | --- | --- |
| PPO + NGU | `python -m experiments.ppo_ngu --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1 --no-wandb` | 0 | `runs/ppo_ngu_seed1_1780566993` | 通过 | 无 |
| PPO + RE3 | `python -m experiments.ppo_re3 --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1 --no-wandb` | 0 | `runs/ppo_re3_seed1_1780567109` | 通过 | 无 |
| PPO + E3B | `python -m experiments.ppo_e3b --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1 --no-wandb` | 0 | `runs/ppo_e3b_seed1_1780567276` | 通过 | 无 |

## 1,000,000 step 训练结果

| 方法 | run_dir | global_step | episodes | best_val_total_reward | summary.json | checkpoints/last.pt | 结果 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| PPO + NGU | `runs/ppo_ngu_1m_seed1_1780568836` | 1000000 | 7547 | 2334.0 | 存在 | 存在 | 通过 |
| PPO + RE3 | `runs/ppo_re3_1m_seed1_1780587129` | 1000000 | 7669 | 1860.5 | 存在 | 存在 | 通过 |
| PPO + E3B | `runs/ppo_e3b_1m_seed1_1780604303` | 1000000 | 7582 | 2016.5 | 存在 | 存在 | 通过 |

## 关键指标摘录

### PPO + NGU

- run_dir：`runs/ppo_ngu_1m_seed1_1780568836`
- summary.json：存在
- train_metrics.csv：存在，rows=7547
- update_metrics.csv：存在，rows=7813
- val_metrics.csv：存在，rows=4004
- test_metrics.csv：存在，rows=32
- checkpoints/last.pt：存在
- video：`videos/ppo_ngu_1m_best_sample.mp4`，存在
- summary：global_step=1000000, episodes=7547, device=cuda, num_actions=7, action_space=SIMPLE_MOVEMENT, best_val_total_reward=2334.0
- update_metrics 最后一行：rollout_int_reward_mean=0.037325600831536576, rollout_int_reward_sum=2.388838453218341, entropy=1.5404702425003052, approx_kl=0.00015522167086601257, dominant_action_frac=0.40625
- train_metrics 最后一条 episode：episode_ext_reward=1690.0, episode_int_reward=0.0, episode_total_reward=1690.0, episode_max_x_pos=1791

### PPO + RE3

- run_dir：`runs/ppo_re3_1m_seed1_1780587129`
- summary.json：存在
- train_metrics.csv：存在，rows=7669
- update_metrics.csv：存在，rows=7813
- val_metrics.csv：存在，rows=4004
- test_metrics.csv：存在，rows=24
- checkpoints/last.pt：存在
- video：`videos/ppo_re3_1m_best_sample.mp4`，存在
- summary：global_step=1000000, episodes=7669, device=cuda, num_actions=7, action_space=SIMPLE_MOVEMENT, best_val_total_reward=1860.5
- update_metrics 最后一行：rollout_int_reward_mean=0.00029708626539104444, rollout_int_reward_sum=0.019013520985026844, entropy=1.5653457641601562, approx_kl=0.00017831195145845413, dominant_action_frac=0.421875
- train_metrics 最后一条 episode：episode_ext_reward=1331.0, episode_int_reward=0.0, episode_total_reward=1331.0, episode_max_x_pos=1433

### PPO + E3B

- run_dir：`runs/ppo_e3b_1m_seed1_1780604303`
- summary.json：存在
- train_metrics.csv：存在，rows=7582
- update_metrics.csv：存在，rows=7813
- val_metrics.csv：存在，rows=4004
- test_metrics.csv：存在，rows=32
- checkpoints/last.pt：存在
- video：`videos/ppo_e3b_1m_best_sample.mp4`，存在
- summary：global_step=1000000, episodes=7582, device=cuda, num_actions=7, action_space=SIMPLE_MOVEMENT, best_val_total_reward=2016.5
- update_metrics 最后一行：rollout_int_reward_mean=0.0005512009677204333, rollout_int_reward_sum=0.03527686193410773, entropy=1.6441340446472168, approx_kl=0.0002306746318936348, dominant_action_frac=0.390625
- train_metrics 最后一条 episode：episode_ext_reward=611.0, episode_int_reward=0.0, episode_total_reward=611.0, episode_max_x_pos=684

## 运行中最小修复

- `rllte/rllte/xplore/reward/pseudo_counts.py`：修复离散动作下 `CrossEntropyLoss(reduction="none")` 返回 1D loss 时的 mask expand 维度错误。
- `rllte/rllte/xplore/reward/e3b.py`：同样修复 E3B 离散动作 loss mask 维度错误。
- 以上修复只影响离散动作 loss mask 的形状适配，不改变 PPO 主训练逻辑或实验范围。

## 验收结论

三个 PPO intrinsic reward 方法均完成 1,000,000 step 训练，并产出 summary、非空训练/更新/验证指标、checkpoint 和视频。
