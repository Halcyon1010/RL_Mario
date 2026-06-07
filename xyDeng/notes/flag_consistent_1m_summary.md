# Flag-consistent 1M training summary

Date: 2026-06-07

This summary compares three methods using training logs that include `episode_flag_get`.

## Runs

| Method | Run | Device | Steps | Episodes | Train `flag_get` count | Train max x_pos | First flag step | First flag episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ICM | `xyDeng_ppo_icm_flag_1m_seed1_1780796302` | CUDA | 1,000,000 | 7798 | 6 | 3161 | 82450 | 577 |
| Disagreement | `xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237` | CPU | 1,000,000 | 7785 | 10 | 3161 | 117353 | 812 |
| PPO | `xyDeng_ppo_extrinsic_flag_1m_seed1_1780806585` | CUDA | 1,000,000 | 7611 | 10 | 3161 | 21765 | 135 |

## Evaluation summary

| Method | Best val total reward | Best val mean max x_pos | Best val single max x_pos |
|---|---:|---:|---:|
| ICM | 945.0 | 1037.0 | 1947 |
| Disagreement | 625.67 | 708.5 | 1666 |
| PPO | 1193.67 | 1328.67 | 2471 |

## Interpretation

All three methods have strict training-time completion evidence under the same `episode_flag_get` logging convention.

For this seed and implementation, pure PPO has the earliest first training completion and the best validation metrics among these flag-consistent runs. ICM also completes the level during training, but its validation performance is below PPO. Disagreement has 10 training completions, but its validation metrics are the weakest in this set; it remains useful for qualitative exploration videos from the earlier World 1 evaluation.

Claim boundary:

- These results are single-seed and should not be presented as a statistically stable ranking.
- Training-time `flag_get` confirms the agent can complete the level during rollouts.
- Evaluation videos still need separate checkpoint search if the report needs a visible completion video.

## Output files

- Metrics CSV: `xyDeng/notes/flag_consistent_1m_metrics.csv`
- Plot directory: `xyDeng/plots/flag_1m_comparison/`
- Reproduction script: `xyDeng/notes/run_xyDeng_icm_and_ppo_flag_1m.ps1`

