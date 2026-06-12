# Final Selected Runs

This directory stores the complete 1,000,000-timestep training runs selected as the final evidence for xyDeng's Mario intrinsic-reward experiments. The layout follows the `chengzhiwei/final_selected_runs` format.

| Method | Complete run directory | Timesteps | Completion evidence | Notes |
|---|---|---:|---|---|
| PPO + ICM | `xyDeng_ppo_icm_1m_cuda_seed1_1780669071/` | 1,000,000 | Inferred: 4 episodes reached `final_x_pos=3161` and `max_x_pos=3161`; original run did not log `episode_flag_get` | Strong training-time completion evidence, but not strict counted flag evidence |
| PPO + Disagreement | `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739/` | 1,000,000 | Inferred: 1 episode reached `final_x_pos=3161` and `max_x_pos=3161`; original run did not log `episode_flag_get` | Best original recorded rollout among sampled checkpoints reached `max_x_pos=2009` |
| PPO + Disagreement World 1 eval | `xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237/` | 1,000,000 | Strict: 10 training episodes logged `episode_flag_get=1` | Used for explicit completion audit and World 1-1 through 1-4 evaluation videos |

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

The World 1 eval run also includes:

- `best_max_x_summary.json`
- `best_single_max_x_summary.json`

Notes:

- These are complete selected runs, so `.pt` checkpoint files are included.
- The current results use seed 1 only and should not be treated as multi-seed statistical evidence.
- See `xyDeng/notes/training_completion_audit.md` for the completion audit.
