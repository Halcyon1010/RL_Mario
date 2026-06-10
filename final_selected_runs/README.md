# Final Selected Runs

This directory stores the three complete 1,000,000-timestep training runs selected as the final evidence for this project. They correspond to the reports, presentation materials, and metrics under `results/metrics/`.

| Method | Complete run directory | Timesteps | Notes |
|---|---|---:|---|
| PPO + NGU | `ppo_ngu_1m_seed1_1780568836/` | 1,000,000 | Highest validation sample mean reward; strong sampled trajectories on both 1-1 and 1-2 |
| PPO + RE3 | `ppo_re3_1m_seed1_1780587129/` | 1,000,000 | Highest single-episode reward in the 1-1 1000-episode sample evaluation |
| PPO + E3B | `ppo_e3b_1m_seed1_1780604303/` | 1,000,000 | More stable mean / median metrics in the 1-2 transfer evaluation |

Each run directory contains:

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

Notes:

- These are complete runs, so `.pt` checkpoint files are included.
- Interrupted runs are not included in this directory.
- The current results use seed 1 only and should not be treated as multi-seed statistical evidence.
