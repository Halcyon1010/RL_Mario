# 最终采用结果

本目录保存本项目最终采用的三个完整 1,000,000 timestep 训练 run。它们对应报告、PPT 和 `results/metrics/` 中使用的最终结果。

| 方法 | 完整 run 目录 | 训练步数 | 说明 |
|---|---|---:|---|
| PPO + NGU | `ppo_ngu_1m_seed1_1780568836/` | 1,000,000 | validation sample mean reward 最高；1-1 与 1-2 都出现较强单局轨迹 |
| PPO + RE3 | `ppo_re3_1m_seed1_1780587129/` | 1,000,000 | 1-1 的 1000 episode sample 中获得最高单局 reward |
| PPO + E3B | `ppo_e3b_1m_seed1_1780604303/` | 1,000,000 | 1-2 transfer 的 mean / median 指标较稳 |

每个 run 目录包含：

- `config.yaml`
- `summary.json`
- `best_summary.json`
- `train_metrics.csv`
- `update_metrics.csv`
- `val_metrics.csv`
- `test_metrics.csv`
- `latest_*_sample.json`
- `latest_*_greedy.json`
- `checkpoints/`

注意：

- 这里保留的是完整 run，因此包含 `.pt` checkpoint 文件。
- 中断 run 未放入本目录。
- 当前结果基于 seed 1，不能作为多 seed 统计显著结论。
