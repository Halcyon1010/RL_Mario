# Patches needed to run on Super Mario Bros (yz)

Two fixes were required for the intrinsic-reward methods to train and to record long evaluations on Mario (discrete action space, GPU). Both are included in this branch.

## 1. Discrete-action mask shape bug in RLeXplore (RIDE, PseudoCounts; also affects ICM, E3B)

For discrete actions, the inverse-dynamics loss `im_loss` is 1-D `[batch]` (`CrossEntropyLoss(reduction="none")`), but the code force-unsqueezes the mask to 2-D and then `expand_as`, which fails:

```
RuntimeError: expand(... {[32, 1]}, size=[32]): the number of sizes provided (1)
must be greater or equal to the number of dimensions in the tensor (2)
```

**Files:** `rllte/rllte/xplore/reward/ride.py`, `rllte/rllte/xplore/reward/pseudo_counts.py`
(the same pattern also exists in `e3b.py` and `icm.py`).

**Fix** — make the mask reshape match the loss rank (works for both 1-D discrete and 2-D continuous losses):

```python
# before
im_mask = mask.unsqueeze(1).expand_as(im_loss)
# after
im_mask = mask.view(mask.shape[0], *([1] * (im_loss.dim() - 1))).expand_as(im_loss)
```

## 2. OOM in checkpoint evaluation when recording many episodes

`mario_rl/evaluate_checkpoint.py` stored **every** episode's video frames in memory (each summary kept its `frames`), so a large `--episodes` (needed to capture a rare level completion) ballooned RAM and OOM-killed the container.

**Fix** — keep only the current best episode's frames:

```python
best_frames = None
...
# drop "frames" from the per-episode summary dict
if best is None or summary[args.save_best_by] > best[args.save_best_by]:
    best = summary
    best_frames = frames          # retain only the best episode's frames
...
imageio.mimsave(output_path, best_frames, fps=video_fps, macro_block_size=1)
```

With this fix, `--episodes 3000+` runs at flat memory, which is how the level-completion clips in `media/` were captured.
