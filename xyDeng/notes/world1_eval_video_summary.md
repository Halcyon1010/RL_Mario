# World 1 evaluation videos

Run:

`xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237`

Configuration:

- Algorithm: PPO + Disagreement intrinsic reward
- Training level: `SuperMarioBros-1-1-v0`
- Evaluation levels: `SuperMarioBros-1-1-v0` through `SuperMarioBros-1-4-v0`
- Training steps: 1,000,000
- Device: CPU
- Evaluation policy: sample
- Search protocol: best episode from 64 sampled episodes per checkpoint, ranked by `(flag_get, max_x_pos, total_reward)`

## Best videos

| Level | Best checkpoint | Saved episode | Flag reached | Max x_pos | Reward | Decision steps | Video |
|---|---:|---:|---:|---:|---:|---:|---|
| 1-1 | `ppo_step_800000.pt` | 9 | 0 | 2473 | 2337 | 402 | `xyDeng/videos/world1_eval/world-1-1_ppo_step_800000_best_of_64.mp4` |
| 1-2 | `best_max_x.pt` | 60 | 0 | 1938 | 1768 | 573 | `xyDeng/videos/world1_eval/world-1-2_best_max_x_best_of_64.mp4` |
| 1-3 | `best_max_x.pt` | 11 | 0 | 775 | 702 | 78 | `xyDeng/videos/world1_eval/world-1-3_best_max_x_best_of_64.mp4` |
| 1-4 | `best_max_x.pt` | 49 | 0 | 1229 | 1147 | 133 | `xyDeng/videos/world1_eval/world-1-4_best_max_x_best_of_64.mp4` |

## Interpretation

No tested episode reached the flag in the 64-episode checkpoint search. The best visual result is World 1-1, where the agent reaches `x_pos=2473`, substantially farther than the earlier 1M Disagreement result. World 1-2 also shows meaningful progress (`x_pos=1938`), while World 1-3 and World 1-4 remain much weaker under a policy trained only on 1-1.

For the project report, use these videos as qualitative evidence of exploration and partial transfer across World 1 levels, not as evidence of solved levels.

## Reproduction commands

Run from:

`C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario`

Example for World 1-1:

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m mario_rl.evaluate_checkpoint --config C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs\xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237\config.yaml --checkpoint C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs\xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237\checkpoints\ppo_step_800000.pt --mode sample --env-id SuperMarioBros-1-1-v0 --episodes 64 --save-best-by flag_get --stop-on-flag --device cpu --fps 30 --playback realtime --output C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\videos\world1_eval\world-1-1_ppo_step_800000_best_of_64.mp4
```

