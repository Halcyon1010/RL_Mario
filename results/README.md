# Results (yz)

- `reports/RUN_REPORT_yz_ride_pseudocounts_rnd.md` — main report for PPO + RIDE / PseudoCounts / RND (1M steps), including level-completion counts and the NGU replication cross-check.
- `plots/` — training/validation/test curves for the three 1M runs (RIDE, PseudoCounts, RND), generated with `mario_rl.utils.plot_runs` (smoothed). Key files: `val_total_reward.png`, `val_max_x_pos.png`, `train_max_x_pos.png`, `train_int_reward.png`.

Underlying complete run directories (with checkpoints) are under `final_selected_runs/`.
