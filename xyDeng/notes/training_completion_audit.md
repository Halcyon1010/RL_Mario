# Training completion audit

Date: 2026-06-07

Question: did ICM and Disagreement complete World 1-1 during training?

## Short answer

Disagreement is strictly confirmed to have completed the level during training in the later World 1 evaluation run, because that run records `episode_flag_get=1`.

ICM cannot be strictly confirmed from the original 1M log because the original run did not record `episode_flag_get`. However, ICM reached `episode_max_x_pos=3161` and `episode_final_x_pos=3161` four times during training, with rewards around 3000. This is strong evidence that it reached the flag/end-of-level range, but it should be reported as an inferred completion rather than a counted `flag_get`.

## Evidence table

| Method | Run | Has `episode_flag_get`? | Counted flag completions | Episodes reaching `max_x_pos >= 3161` | Conclusion |
|---|---|---:|---:|---:|---|
| ICM | `xyDeng_ppo_icm_1m_cuda_seed1_1780669071` | No | Not available | 4 | Strong inferred completion, not strictly logged |
| Disagreement | `xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739` | No | Not available | 1 | Strong inferred completion, not strictly logged |
| Disagreement | `xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237` | Yes | 10 | 10 | Strictly confirmed completion |

## ICM inferred completion episodes

| global_step | episode | decision_steps | reward | final_x_pos | max_x_pos |
|---:|---:|---:|---:|---:|---:|
| 59681 | 397 | 544 | 3013.0 | 3161 | 3161 |
| 107081 | 747 | 441 | 3033.0 | 3161 | 3161 |
| 412919 | 3062 | 585 | 2999.0 | 3161 | 3161 |
| 607218 | 4371 | 535 | 3015.0 | 3161 | 3161 |

## Disagreement strictly confirmed completion episodes

| global_step | episode | decision_steps | reward | final_x_pos | max_x_pos | episode_flag_get |
|---:|---:|---:|---:|---:|---:|---:|
| 117353 | 812 | 399 | 3042.0 | 3161 | 3161 | 1 |
| 129557 | 915 | 534 | 3015.0 | 3161 | 3161 | 1 |
| 173179 | 1258 | 621 | 2991.0 | 3161 | 3161 | 1 |
| 199763 | 1462 | 478 | 3026.0 | 3161 | 3161 | 1 |
| 252593 | 1872 | 612 | 2999.0 | 3161 | 3161 | 1 |
| 296030 | 2221 | 488 | 3024.0 | 3161 | 3161 | 1 |
| 523777 | 4050 | 616 | 2992.0 | 3161 | 3161 | 1 |
| 855339 | 6655 | 590 | 3003.0 | 3161 | 3161 | 1 |
| 935442 | 7293 | 541 | 3013.0 | 3161 | 3161 | 1 |
| 987333 | 7678 | 507 | 3020.0 | 3161 | 3161 | 1 |

## Recommended report wording

Use this conservative wording:

> During training, Disagreement produced 10 explicitly logged flag completions (`episode_flag_get=1`). ICM did not log `flag_get` in the original 1M run, but reached the terminal x-position (`x_pos=3161`) four times with high rewards, so we treat ICM as having strong inferred completion evidence rather than strict counted flag completions.

Avoid claiming that the original ICM and original Disagreement runs both have explicit `flag_get` evidence. They do not.

