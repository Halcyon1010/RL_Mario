# Media (yz)

Best gameplay videos from the 1M runs, recorded on `SuperMarioBros-1-1-v0` by sampling from the best checkpoint and saving the farthest-progress episode (`--save-best-by max_x_pos`).

| File | Method | What it shows |
|---|---|---|
| `ppo_ride_1_1_clear.mp4` | RIDE | **Full level completion** (reaches the flag, x=3161, reward ≈ 3012). Captured from `last.pt` over 3000 sampled episodes. |
| `ppo_ride_1_1_best_sample.mp4` | RIDE | Best of a 10-episode sample (mid-level progress). |
| `ppo_pseudocounts_1_1_best.mp4` | PseudoCounts | Best sampled episode (~78% of the level). |
| `ppo_pseudocounts_1_1_best_sample.mp4` | PseudoCounts | Best of a 10-episode sample. |
| `ppo_rnd_1_1_best_sample.mp4` | RND | Best of a 10-episode sample (~plateau progress). |
| `ppo_ngu_1_1_clear_replication.mp4` | NGU (replication) | **Full level completion by PPO+NGU** — reproduces the NGU completion result from the `chengzhiwei` branch (x=3161, reward ≈ 2997). |

Notes:

- Level completions are rare events (~0.1% of episodes even for the trained policy), so the clear clips were obtained by sampling thousands of episodes and keeping the farthest one.
- The NGU clip is included only as a cross-check/replication of `chengzhiwei`'s NGU result; NGU itself is owned by that branch.
