# Final Selected Runs: World 1-1 Only

This directory contains only the selected 1,000,000-timestep runs that train and evaluate on `SuperMarioBros-1-1-v0`.

It is a non-destructive 1-1-only subset. Other existing results in the repository are intentionally kept.

| Method | Complete run directory | Timesteps | Train env | Val/Test env | Completion evidence |
|---|---|---:|---|---|---|
| PPO + ICM | `xyDeng_ppo_icm_1m_cuda_seed1_1780669071/` | 1,000,000 | `SuperMarioBros-1-1-v0` | `SuperMarioBros-1-1-v0` | Inferred: 4 episodes reached `final_x_pos=3161` and `max_x_pos=3161` |
| PPO + Disagreement | `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739/` | 1,000,000 | `SuperMarioBros-1-1-v0` | `SuperMarioBros-1-1-v0` | Inferred: 1 episode reached `final_x_pos=3161` and `max_x_pos=3161` |

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

- These are complete selected runs, so `.pt` checkpoint files are included.
- The selected logs do not include `episode_flag_get`; completion should be reported as inferred from terminal x-position.
- The broader `final_selected_runs/` and `xyDeng/` result folders are left unchanged.
