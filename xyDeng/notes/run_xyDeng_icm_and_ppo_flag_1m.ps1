$ErrorActionPreference = "Stop"

$Python = "C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe"
$Project = "C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario"
$RunDir = "C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs"

Set-Location $Project

& $Python -m experiments.ppo_icm `
  --total-timesteps 1000000 `
  --eval-interval 50000 `
  --eval-episodes 3 `
  --test-episodes 3 `
  --save-interval 25000 `
  --device cuda `
  --experiment-name xyDeng_ppo_icm_flag_1m `
  --run-dir $RunDir `
  --no-wandb

& $Python -m experiments.ppo_extrinsic `
  --total-timesteps 1000000 `
  --eval-interval 50000 `
  --eval-episodes 3 `
  --test-episodes 3 `
  --save-interval 25000 `
  --device cuda `
  --experiment-name xyDeng_ppo_extrinsic_flag_1m `
  --run-dir $RunDir `
  --no-wandb
