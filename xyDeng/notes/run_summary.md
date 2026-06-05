# xyDeng 1M Mario Runs Summary

Date: 2026-06-06

Environment:

- `SuperMarioBros-1-1-v0`
- `SIMPLE_MOVEMENT`
- frame skip: 4
- observation: `4 x 84 x 84`
- seed: 1

## Device Benchmark

10k-step benchmark:

| Method | Device | Seconds | Seconds / 1k steps |
|---|---:|---:|---:|
| ICM | CPU | 118.84 | 11.88 |
| ICM | CUDA | 85.41 | 8.54 |
| Disagreement | CPU | 120.14 | 12.01 |
| Disagreement | CUDA | 134.27 | 13.43 |

Decision:

- ICM 1M used CUDA.
- Disagreement 1M used CPU.

## 1M Training Runs

| Method | Run | Device | Episodes | Best eval step | Best val sample mean max_x | Best test sample mean max_x | Train max_x |
|---|---|---:|---:|---:|---:|---:|---:|
| ICM | `xyDeng_ppo_icm_1m_cuda_seed1_1780669071` | CUDA | 7480 | 900000 | 1237.2 | 785.0 | 3161 |
| Disagreement | `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739` | CPU | 7994 | 950000 | 1140.8 | 784.6 | 3161 |

Both methods reached `episode_max_x_pos=3161` during training. The logging schema does not include `flag_get`, so this should be described as "reached the far end / likely flag-reaching range" rather than a strict counted flag completion.

## Recorded Videos

Best validation checkpoint videos:

| Method | Checkpoint | Video | Best sampled max_x | Notes |
|---|---|---|---:|---|
| ICM | `best.pt` | `xyDeng_ppo_icm_1m_cuda_best_sample.mp4` | 1418 | no visible completion |
| Disagreement | `best.pt` | `xyDeng_ppo_disagreement_1m_cpu_best_sample.mp4` | 1678 | no visible completion |

Additional checkpoint search videos:

| Method | Checkpoint | Best sampled max_x | Video |
|---|---|---:|---|
| ICM | `ppo_step_400000.pt` | 1432 | `xyDeng_ppo_icm_1m_cuda_seed1_1780669071_ppo_step_400000_sample20.mp4` |
| ICM | `ppo_step_800000.pt` | 1435 | `xyDeng_ppo_icm_1m_cuda_seed1_1780669071_ppo_step_800000_sample20.mp4` |
| ICM | `last.pt` | 1418 | `xyDeng_ppo_icm_1m_cuda_seed1_1780669071_last_sample20.mp4` |
| Disagreement | `ppo_step_400000.pt` | 1671 | `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739_ppo_step_400000_sample20.mp4` |
| Disagreement | `ppo_step_800000.pt` | 1974 | `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739_ppo_step_800000_sample20.mp4` |
| Disagreement | `last.pt` | 2009 | `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739_last_sample20.mp4` |

Best recorded video overall:

- `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739_last_sample20.mp4`
- checkpoint: Disagreement `last.pt`
- sampled best `max_x_pos`: 2009
- no full level completion observed in the recorded videos.

## Output Locations

- Runs: `xyDeng/runs/`
- Videos: `xyDeng/videos/`
- Plots: `xyDeng/plots/1m_comparison/`
- Metrics summary CSV: `xyDeng/notes/one_million_run_metrics.csv`
- Device benchmark CSV: `xyDeng/notes/device_benchmark_10k.csv`

## Interpretation

For report wording:

- Strong evidence: both ICM and Disagreement produced far-reaching training episodes, with `max_x_pos=3161`.
- Reproducible video evidence: Disagreement produced the best recorded rollout among sampled checkpoints, reaching `max_x_pos=2009`.
- Conservative claim boundary: no recorded sample video fully completes the level, and the logs do not include an explicit `flag_get` column.
