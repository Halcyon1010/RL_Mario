# Mario PPO Intrinsic Reward Experiments

This branch contains the Mario reinforcement learning experiments comparing three PPO-based intrinsic reward methods:

- PPO + NGU
- PPO + RE3
- PPO + E3B

The final comparison uses 1,000,000 training timesteps on `SuperMarioBros-1-1-v0` with seed 1.

Curated evidence:

- source code and configs: `configs/`, `experiments/`, `mario_rl/`
- patched local RLLTE dependency: `rllte/`
- final metrics and reports: `results/`
- representative videos: `media/`
- presentation deck and embedded GIF assets: `presentation/`
- final selected complete runs: `final_selected_runs/`

Important boundary:

- Full non-selected training artifacts are not committed.
- Checkpoints are committed only for the three final selected runs under `final_selected_runs/`.
- Interrupted runs are not used as final evidence.
- Current results use a single seed, so they should be reported as observed results under this setting, not as statistically significant conclusions.

Quick smoke commands:

```powershell
docker compose build
docker compose run --rm mario-rl python -m mario_rl.check_env
docker compose run --rm mario-rl python -m mario_rl.check_rllte
docker compose run --rm mario-rl python -m experiments.ppo_ngu --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
docker compose run --rm mario-rl python -m experiments.ppo_re3 --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
docker compose run --rm mario-rl python -m experiments.ppo_e3b --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

# Mario RL Docker Environment

This workspace contains the first-stage Docker environment for the Super Mario Bros reinforcement learning project.

It includes both:

- the legacy Gym stack used by `gym-super-mario-bros`
- the RLeXplore/RLLTE stack used by `from rllte.xplore.reward import ...`

## Optional: Use Local RLLTE Source

To develop against the open-source package directly, clone RLLTE into this folder:

```powershell
cd D:\Mario
git clone https://github.com/RLE-Foundation/rllte.git
docker compose build
```

The Dockerfile automatically installs `./rllte` in editable mode when the folder exists. This lets project code import intrinsic reward algorithms from the local source package:

```python
from rllte.xplore.reward import RND, ICM, RE3
```

The helper in `mario_rl.rewards.rllte_intrinsic` can build modules by name, such as `rnd`, `icm`, or `re3`.

## Build

```powershell
docker compose build
```

## Check The Environment

```powershell
docker compose run --rm mario-rl python -m mario_rl.check_env
```

The check creates `SuperMarioBros-1-1-v0`, wraps it with `RIGHT_ONLY`, runs random actions, and prints reward plus `x_pos`.

## Check RLeXplore Imports

```powershell
docker compose run --rm mario-rl python -c "from rllte.xplore.reward import RND, ICM, RE3; print('rlexplore=ok')"
```

## Run PPO Smoke Test

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_smoke
```

This runs a short PPO-only training test and writes logs under `runs/ppo_smoke_*`.

## Run PPO Extrinsic Baseline

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_extrinsic
```

This runs the PPO baseline with external reward only and writes `train_metrics.csv`, `update_metrics.csv`, `val_metrics.csv`, `test_metrics.csv`, `summary.json`, and checkpoints under `runs/ppo_extrinsic_*`.

Validation runs every `eval_interval` steps. The best model is selected by validation mean total reward:

- `checkpoints/best.pt`: saved whenever validation mean total reward improves
- `checkpoints/last.pt`: latest model
- `checkpoints/ppo_step_<step>.pt`: periodic snapshots controlled by `save_interval`

Whenever validation improves, the current best model is immediately evaluated on the test environments and written to `test_metrics.csv`. Validation and test logs both keep external, intrinsic, and total rewards separately.

You can override common settings from the command line:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_extrinsic --total-timesteps 50000 --eval-interval 5000 --save-interval 50000
```

Useful overrides:

- `--total-timesteps 50000`
- `--eval-interval 5000`
- `--eval-episodes 5`
- `--test-episodes 5`
- `--save-interval 50000`
- `--device cuda`
- `--seed 2`
- `--experiment-name ppo_extrinsic_50k`

## Run PPO + RND

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_rnd --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1
```

For a longer run:

```powershell
docker compose run --rm mario-rl python -m experiments.ppo_rnd --total-timesteps 50000 --eval-interval 5000 --eval-episodes 3 --test-episodes 3 --track-wandb --experiment-name ppo_rnd_50k
```

## Plot Experiment Curves

Plot one run:

```powershell
docker compose run --rm mario-rl python -m mario_rl.utils.plot_runs runs/ppo_smoke_seed1_1780481632 --output-dir results/plots/ppo_smoke
```

Plot all runs under `runs/`:

```powershell
docker compose run --rm mario-rl python -m mario_rl.utils.plot_runs runs --output-dir results/plots/all_runs
```

The plotting script creates curves for rewards, x-position, episode steps, visited bins, PPO losses, entropy, and KL.

## Record A Mario Demo Video

```powershell
docker compose run --rm mario-rl python -m mario_rl.record_random --output videos/mario_demo.mp4
```

The video will be saved to `videos/mario_demo.mp4`. This only exports gameplay frames; it does not train or run any RL algorithm.

The default demo uses a scripted action timeline: stand still, walk right, jump right, run right, run-jump right, pause, walk left, jump left, and recover to the right. Formal training will still use the smaller `RIGHT_ONLY` action set described in the project plan.

## Evaluate A Checkpoint And Record Video

```powershell
docker compose run --rm mario-rl python -m mario_rl.evaluate_checkpoint --config runs/ppo_extrinsic_seed1_1780482862/config.yaml --checkpoint runs/ppo_extrinsic_seed1_1780482862/checkpoints/ppo_step_50000.pt --mode sample --output videos/ppo_extrinsic_50k_sample.mp4
```

Use `--mode greedy` to record the deterministic policy. The script prints external, intrinsic, and total reward plus x-position metrics.

## Record Separate Action Videos And GIFs

```powershell
docker compose run --rm mario-rl python -m mario_rl.record_action_gallery --output-dir videos/action_gallery
```

This creates one `.mp4` and one `.gif` for each action:

- `noop`
- `right`
- `right_jump`
- `right_run`
- `right_run_jump`
- `left`
- `left_jump`

## Open A Shell

```powershell
docker compose run --rm mario-rl
```
