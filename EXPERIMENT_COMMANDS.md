# Mario PPO Intrinsic Reward Commands

All commands below should be run from `D:\Mario`.

Common settings:

```powershell
cd D:\Mario
```

## Docker Setup

After receiving and extracting the project zip, build the Docker image:

```powershell
cd D:\Mario
docker compose build
```

Check the Mario environment:

```powershell
docker compose run --rm mario-rl python -m mario_rl.check_env
```

Check the local RLLTE/RLeXplore package:

```powershell
docker compose run --rm mario-rl python -m mario_rl.check_rllte
```

Optional: check CUDA availability inside Docker:

```powershell
docker compose run --rm mario-rl python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## WandB Notes

WandB is optional. Local CSV/JSON logs are always saved under `runs/`, so experiments can be plotted without WandB.

To train without WandB, omit `--track-wandb`:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_rnd --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_rnd_50k
```

To train with WandB, each member should use their own key:

```powershell
$env:WANDB_API_KEY="your_own_wandb_key"
docker compose run --rm mario-rl python -m experiments.ppo_rnd --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_rnd_50k
```

Do not share `secrets/wandb.json`.

The default training setup uses:

- environment: `SuperMarioBros-1-1-v0`
- action space: `SIMPLE_MOVEMENT`
- visual input: `4 x 84 x 84`
- validation modes: `greedy` and `sample`
- best checkpoint selection: validation `sample` mean total reward
- checkpoints: `best.pt`, `last.pt`, and periodic `ppo_step_<step>.pt`

For a quick smoke test, use `--total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1`.

For a standard 50k run, use `--total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb`.

If you do not want WandB, remove `--track-wandb` from the command.

## PPO Baseline

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_extrinsic --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_extrinsic --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_extrinsic_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_extrinsic --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_extrinsic_50k
```

Evaluate best checkpoint and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + RND

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_rnd --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_rnd --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_rnd_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_rnd --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_rnd_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + ICM

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_icm --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_icm --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_icm_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_icm --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_icm_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + RIDE

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_ride --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_ride --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_ride_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_ride --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_ride_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + NGU

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_ngu --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_ngu --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_ngu_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_ngu --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_ngu_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + RE3

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_re3 --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_re3 --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_re3_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_re3 --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_re3_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + E3B

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_e3b --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_e3b --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_e3b_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_e3b --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_e3b_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + PseudoCounts

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_pseudocounts --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_pseudocounts --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_pseudocounts_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_pseudocounts --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_pseudocounts_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## PPO + Disagreement

Smoke test:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_disagreement --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

Train:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_disagreement --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_disagreement_50k
```

Train without WandB:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_disagreement --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --experiment-name ppo_disagreement_50k
```

Evaluate and record video:

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/<RUN_DIR>/config.yaml --checkpoint runs/<RUN_DIR>/checkpoints/best.pt --mode sample --episodes 20 --save-best-by max_x_pos --output videos/<RUN_DIR>_best_sample.mp4
```

## Plot Results

Plot all runs:

```powershell
docker compose run --rm mario-rl python -m mario_rl.utils.plot_runs runs --output-dir results/plots/all_runs --smooth 10
```

Plot selected runs by passing each run directory:

```powershell
docker compose run --rm mario-rl python -m mario_rl.utils.plot_runs runs/<RUN_A> runs/<RUN_B> --output-dir results/plots/selected --smooth 10
```
