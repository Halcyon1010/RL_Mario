# Final Selected Runs (yz)

This directory stores the complete 1,000,000-timestep training runs selected as the final evidence for the **yz** members' methods (PPO + RIDE / PseudoCounts / RND). They correspond to the report and plots under `results/`.

| Method | Complete run directory | Timesteps | Notes |
|---|---|---:|---|
| PPO + RIDE | `ppo_ride_1m_seed1_1780648820/` | 1,000,000 | Best method on SMB-1-1: highest validation sample reward (1611) and farthest progress (~54% of level). Earliest level completion (step 95k). |
| PPO + PseudoCounts | `ppo_pseudocounts_1m_seed1_1780648820/` | 1,000,000 | Second-best validation reward (1260). Most level completions during training (8). |
| PPO + RND | `ppo_rnd_1m_seed1_1780648822/` | 1,000,000 | Robust global count-based baseline (best val 1136). Plateaus around ~39% of the level. |

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

- These are complete runs, so `.pt` checkpoint files are included (checkpoints saved every 100k steps plus `best.pt` / `last.pt`).
- All runs use **seed 1 only** and should not be treated as multi-seed statistical evidence.
- Environment: `SuperMarioBros-1-1-v0`, `SIMPLE_MOVEMENT`, 4×84×84 frames, extrinsic + intrinsic reward, GPU (RTX 5070).
