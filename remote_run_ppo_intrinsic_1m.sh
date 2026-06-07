#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Mario/mario_project"
source "$HOME/Mario/rllte-venv/bin/activate"

mkdir -p long_logs videos
trap 'code=$?; echo "===== RUNNER EXIT code=${code} $(date -Iseconds) =====" | tee -a long_logs/master.log' EXIT

run_and_record() {
  local module="$1"
  local name="$2"
  local config="$3"
  local log="long_logs/${name}.log"
  local video="videos/${name}_best_sample.mp4"

  echo "===== LONG ${name} START $(date -Iseconds) =====" | tee -a long_logs/master.log
  python -u -m "experiments.${module}" \
    --total-timesteps 1000000 \
    --no-wandb \
    --device cuda \
    --experiment-name "${name}" \
    2>&1 | tee "${log}"

  local run_dir
  run_dir="$(grep -Eo 'run_dir=[^[:space:]]+' "${log}" | tail -1 | cut -d= -f2-)"
  if [[ -z "${run_dir}" ]]; then
    echo "run_dir missing for ${name}" | tee -a long_logs/master.log
    exit 1
  fi

  echo "===== VIDEO ${name} run_dir=${run_dir} START $(date -Iseconds) =====" | tee -a long_logs/master.log
  python -u -m mario_rl.evaluate_checkpoint \
    --config "${run_dir}/config.yaml" \
    --checkpoint "${run_dir}/checkpoints/best.pt" \
    --output "${video}" \
    --mode sample \
    --episodes 20 \
    --save-best-by max_x_pos \
    --device cuda \
    2>&1 | tee "long_logs/${name}_video.log"

  echo "method=${name} run_dir=${run_dir} video=${video}" >> long_logs/run_dirs.txt
  echo "===== LONG ${name} DONE $(date -Iseconds) =====" | tee -a long_logs/master.log
}

run_and_record ppo_ngu ppo_ngu_1m configs/ppo_ngu.yaml
run_and_record ppo_re3 ppo_re3_1m configs/ppo_re3.yaml
run_and_record ppo_e3b ppo_e3b_1m configs/ppo_e3b.yaml

echo "ALL_DONE $(date -Iseconds)" | tee -a long_logs/master.log
