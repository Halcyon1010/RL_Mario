# xyDeng Mario RL presentation

Final deck:

- `xyDeng_Mario_RL_ICM_Disagreement_1_1.pptx`

This deck follows the layout style of the `chengzhiwei` branch presentation:
light gray background, white rounded panels, method-color labels, KPI cards,
result tables, and evidence footers.

Scope:

- Training/evaluation setting: `SuperMarioBros-1-1-v0`
- Methods: PPO + ICM and PPO + Disagreement
- Timesteps: 1,000,000
- Seed: 1
- Main result sources:
  - `final_selected_runs_1_1/`
  - `xyDeng/notes/run_summary.md`
  - `xyDeng/notes/training_completion_audit.md`
  - `xyDeng/notes/one_million_run_metrics.csv`
  - `xyDeng/plots/1m_comparison/`
  - `xyDeng/videos/`

The deck reports completion evidence conservatively. The selected 1-1 runs
reach terminal `x_pos=3161` during training, while the saved sampled videos are
used as rollout visualizations rather than strict full-clear videos.

The build script is `build_xyDeng_mario_presentation.mjs` and uses
`@oai/artifact-tool` to export an editable PowerPoint deck.
