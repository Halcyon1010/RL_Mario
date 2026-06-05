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
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m experiments.ppo_icm --total-timesteps 1000000 --eval-interval 50000 --eval-episodes 5 --test-episodes 5 --save-interval 100000 --device cuda --experiment-name xyDeng_ppo_icm_1m --run-dir C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs --no-wandb
```

PPO + Disagreement:

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m experiments.ppo_disagreement --total-timesteps 1000000 --eval-interval 50000 --eval-episodes 5 --test-episodes 5 --save-interval 100000 --device cuda --experiment-name xyDeng_ppo_disagreement_1m --run-dir C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs --no-wandb
```

## Record Result Videos

Replace `<RUN_DIR>` with the generated run folder name under `xyDeng\runs`.

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m mario_rl.evaluate_checkpoint --config C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs\<RUN_DIR>\config.yaml --checkpoint C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs\<RUN_DIR>\checkpoints\best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\videos\<RUN_DIR>_best_sample.mp4 --device cuda
```

## Plot Curves

```powershell
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m mario_rl.utils.plot_runs C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs --output-dir C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\plots --smooth 10
```

## What To Commit

Commit CSV/JSON metrics, plots, videos, and notes.

Do not commit model checkpoints unless explicitly needed.
