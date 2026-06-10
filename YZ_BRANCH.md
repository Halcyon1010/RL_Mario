# yz branch

This branch holds the **yz** members' work for the PPO + intrinsic-reward Mario project: methods **PPO + RIDE / PseudoCounts / RND** (NGU / RE3 / E3B are on the `chengzhiwei` branch).

## Contents

- **Project code** — `configs/`, `experiments/`, `mario_rl/`, `rllte/` (with the fixes in `docs/PATCHES.md`), plus `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md`, `EXPERIMENT_COMMANDS.md`.
- **`final_selected_runs/`** — the three complete 1M-step runs (RIDE, PseudoCounts, RND) with metrics and checkpoints.
- **`media/`** — best gameplay videos, including a confirmed **RIDE level completion** and an **NGU completion** (replication cross-check of `chengzhiwei`).
- **`results/`** — `reports/RUN_REPORT_yz_ride_pseudocounts_rnd.md` and the smoothed training curves in `results/plots/`.
- **`docs/PATCHES.md`** — the two code fixes required to run these methods on Mario (discrete-action mask bug; evaluation OOM bug).

## Headline

| Method | 1M best-val reward | level completions (training) | first completion |
|---|---:|---:|---:|
| RIDE | 1611 | 7 | 95k (earliest) |
| PseudoCounts | 1260 | 8 | 136k |
| RND | 1136 | 5 | 214k |

Ranking **RIDE > PseudoCounts > RND**, consistent with the RLeXplore paper's SMB results. Single seed (seed 1).

See `results/reports/RUN_REPORT_yz_ride_pseudocounts_rnd.md` for details.
