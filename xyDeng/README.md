# xyDeng Mario Intrinsic Reward Runs

This folder is for Xingyu Deng's Mario RL experiment outputs.

Training code lives in:

`C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario`

Shared result repository lives in:

`C:\Users\250010109\data\RL_projects\RL_Mario`

## Environment

Use the local CUDA conda environment:

```powershell
$env:PYTHONPATH="C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario;C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario\rllte"
```

Python:

```powershell
C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe
```

## 1M Training Commands

Run these from the training-code folder:

```powershell
cd C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario
```

PPO + ICM:

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m experiments.ppo_icm --total-timesteps 1000000 --eval-interval 50000 --eval-episodes 5 --test-episodes 5 --save-interval 25000 --device cuda --experiment-name xyDeng_ppo_icm_1m --run-dir C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs --no-wandb
```

PPO + Disagreement:

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m experiments.ppo_disagreement --total-timesteps 1000000 --eval-interval 50000 --eval-episodes 5 --test-episodes 5 --save-interval 25000 --device cpu --experiment-name xyDeng_ppo_disagreement_1m --run-dir C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs --no-wandb
```

## Record Result Videos

Replace `<RUN_DIR>` with the generated run folder name under `xyDeng\runs`.

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m mario_rl.evaluate_checkpoint --config C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs\<RUN_DIR>\config.yaml --checkpoint C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs\<RUN_DIR>\checkpoints\best_single_max_x.pt --mode sample --episodes 256 --save-best-by flag_get --stop-on-flag --output C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\videos\<RUN_DIR>_flag_search.mp4 --device cuda
```

For older runs that do not have `best_single_max_x.pt`, use `best.pt`, `last.pt`, or a saved `ppo_step_<step>.pt` checkpoint instead.

## Plot Curves

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m mario_rl.utils.plot_runs C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs --output-dir C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\plots --smooth 10
```

## World 1 Evaluation Videos

Latest run:

`xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237`

This run trained PPO + Disagreement for 1,000,000 steps on `SuperMarioBros-1-1-v0`, then evaluated `SuperMarioBros-1-1-v0` through `SuperMarioBros-1-4-v0`.

Best-of-64 sample videos:

| Level | Max x_pos | Flag reached | Video |
|---|---:|---:|---|
| 1-1 | 2473 | 0 | `xyDeng/videos/world1_eval/world-1-1_ppo_step_800000_best_of_64_terminal_hold.mp4` |
| 1-2 | 1938 | 0 | `xyDeng/videos/world1_eval/world-1-2_best_max_x_best_of_64.mp4` |
| 1-3 | 775 | 0 | `xyDeng/videos/world1_eval/world-1-3_best_max_x_best_of_64.mp4` |
| 1-4 | 1229 | 0 | `xyDeng/videos/world1_eval/world-1-4_best_max_x_best_of_64.mp4` |

Full notes:

`xyDeng/notes/world1_eval_video_summary.md`

## Training Completion Audit

Completion evidence is summarized in:

`xyDeng/notes/training_completion_audit.md`

Key conclusion:

- Disagreement has strict training-time completion evidence in `xyDeng_ppo_disagreement_world1_eval_1m_seed1_1780716237`: 10 episodes with `episode_flag_get=1`.
- ICM did not record `episode_flag_get` in the original 1M run, but reached `episode_final_x_pos=3161` and `episode_max_x_pos=3161` four times. Treat this as strong inferred completion evidence, not strict counted flag evidence.

## What To Commit

Commit CSV/JSON metrics, plots, videos, and notes.

Do not commit model checkpoints unless explicitly needed.
