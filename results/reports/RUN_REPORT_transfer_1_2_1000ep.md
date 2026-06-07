# PPO Intrinsic Transfer Report: 1-1 Trained Models on 1-2

- Generated: 2026-06-05
- Evaluation environment: `SuperMarioBros-1-2-v0`
- Source checkpoints: best checkpoints from 1,000,000-step training on `SuperMarioBros-1-1-v0`
- Methods: PPO + NGU, PPO + RE3, PPO + E3B
- Evaluation mode: `sample`
- Episodes per method: 1000
- Best-video selection metric: `max_x_pos`

## Summary

The 1-1-trained policies can transfer partially to 1-2, but none of the 1000 sampled episodes completed the 1-2 level. The strongest 1-2 result was PPO + NGU, which reached `max_x_pos=2272` in episode 971.

| Method | Best episode | Best max_x_pos | Best ext_reward | Decision steps | Video |
| --- | ---: | ---: | ---: | ---: | --- |
| PPO + NGU | 971 | 2272 | 2070.00 | 732 | `outputs/ppo_ngu_1m_transfer_1_2_best_sample_1000ep_final.mp4` |
| PPO + RE3 | 261 | 1940 | 1747.00 | 691 | `outputs/ppo_re3_1m_transfer_1_2_best_sample_1000ep_final.mp4` |
| PPO + E3B | 617 | 1955 | 1810.00 | 441 | `outputs/ppo_e3b_1m_transfer_1_2_best_sample_1000ep_final.mp4` |

## Interpretation

NGU produced the best transfer episode on 1-2. RE3 and E3B were close to each other around the `x_pos=1940-1955` range, while NGU later found a stronger trajectory reaching `x_pos=2272`.

Compared with the 10-episode quick transfer baseline:

| Method | 10-episode best max_x_pos | 1000-episode best max_x_pos | Improvement |
| --- | ---: | ---: | ---: |
| PPO + NGU | 887 | 2272 | +1385 |
| PPO + RE3 | 957 | 1940 | +983 |
| PPO + E3B | 875 | 1955 | +1080 |

This indicates the trained policies are stochastic enough that repeated sampling can find much better trajectories than short evaluation runs. For reporting, the 1000-episode best sample is more representative of "best observed transfer behavior", while the 10-episode result is only a quick smoke-test baseline.

## Metric Notes

The video search used:

```text
--mode sample --episodes 1000 --save-best-by max_x_pos
```

Therefore the selected final videos are the episodes that got farthest horizontally, not necessarily the episodes with the highest external reward.

The video recording script supports only these best-selection metrics:

```text
max_x_pos
total_reward
```

In the current evaluation script, `total_reward` is equal to external environment reward (`ext_reward`). It does not compute or rank by intrinsic reward during video evaluation.

## Output Files

Local final videos:

```text
/Users/chengzhiwei/Documents/Mario/outputs/ppo_ngu_1m_transfer_1_2_best_sample_1000ep_final.mp4
/Users/chengzhiwei/Documents/Mario/outputs/ppo_re3_1m_transfer_1_2_best_sample_1000ep_final.mp4
/Users/chengzhiwei/Documents/Mario/outputs/ppo_e3b_1m_transfer_1_2_best_sample_1000ep_final.mp4
```

Remote WSL logs:

```text
/home/user/Mario/mario_project/long_logs/ppo_ngu_1m_transfer_1_2_1000ep.log
/home/user/Mario/mario_project/long_logs/ppo_re3_1m_transfer_1_2_1000ep.log
/home/user/Mario/mario_project/long_logs/ppo_e3b_1m_transfer_1_2_1000ep.log
```

## Relation to 1-1 Report

The main 1-1 training report is:

```text
/Users/chengzhiwei/Documents/Mario/outputs/RUN_REPORT_ppo_ngu_re3_e3b.md
```

That report covers training completion, run directories, summaries, checkpoints, and original 1-1 validation/test outputs. This 1-2 report covers only cross-level transfer evaluation.
