# Run Report — yz (PPO + RIDE / PseudoCounts / RND)

**Environment:** SuperMarioBros-1-1-v0 · SIMPLE_MOVEMENT · 4×84×84 · extrinsic + intrinsic reward
**Backbone:** PPO (RLeXplore intrinsic rewards) · **seed 1** · GPU (RTX 5070, torch 2.8 + CUDA 12.8)
**Budget:** 1,000,000 timesteps per method (plus 50k pilot runs not reported here)

## 1. Method selection

The group covers the 8 RLeXplore intrinsic rewards. NGU / RE3 / E3B are owned by the `chengzhiwei` branch. From the remaining pool (ICM, RND, Disagreement, PseudoCounts, RIDE), **RIDE, PseudoCounts, RND** were selected as the three most likely to perform on Super Mario Bros, based on the paper's SMB results (RIDE is a top single objective with reported 100% completion; PseudoCounts has the highest SMB asymptotic return; RND is a robust global count-based baseline).

## 2. Headline results (1M steps)

| Rank | Method | Best val reward (sample) | Best val max-x | % of level | Best step |
|---|---|---:|---:|---:|---:|
| 1 | **RIDE** | 1611 | 1706 | ~54% | 100k |
| 2 | **PseudoCounts** | 1260 | 1363 | ~43% | 850k |
| 3 | **RND** | 1136 | 1229 | ~39% | 800k |

Final ordering **RIDE > PseudoCounts > RND** matches the paper's SMB ranking. RND saturates around 1/3–2/5 of the level regardless of budget, while the curiosity/dynamics-based RIDE keeps climbing.

## 3. Level completions ("通关")

All three methods **actually finish the level** during training (reach the flagpole at x≈3161, reward ≈ 3000, not a timeout):

| Method | Completions @1M (training) | First completion step |
|---|---:|---:|
| PseudoCounts | 8 | 136,297 |
| RIDE | 7 | **95,136 (earliest)** |
| RND | 5 | 214,099 |

Completion videos: `media/ppo_ride_1_1_clear.mp4` (RIDE, confirmed clear). Completions are rare (~0.1% of episodes), so the clip was captured by sampling 3000 episodes from `last.pt` and keeping the farthest one.

### NGU replication (cross-check of `chengzhiwei`)
We also ran PPO + NGU for 1M steps to verify the NGU completion result: best val 987 / max-x 1084, **3 level completions** (first at step 356,875). Clip: `media/ppo_ngu_1_1_clear_replication.mp4`. This reproduces that PPO + NGU solves SMB-1-1.

## 4. Reproduction

See `EXPERIMENT_COMMANDS.md`. Example (1M, no W&B):

```
docker compose run --rm mario-rl python -m experiments.ppo_ride \
  --total-timesteps 1000000 --eval-interval 50000 --eval-episodes 3 \
  --test-episodes 3 --save-interval 100000 --experiment-name ppo_ride_1M
```

Two fixes were required to run these methods on Mario; see `docs/PATCHES.md`.

## 5. Caveats

- **Single seed (seed 1).** Not multi-seed statistical evidence; add seeds 2–3 for error bars.
- Completion counts are from training rollouts (stochastic policy); per-checkpoint evaluation completes less often.
- The project trains on the singleton level 1-1; episodic methods (RIDE, PseudoCounts) would be favoured further on randomized stages.
