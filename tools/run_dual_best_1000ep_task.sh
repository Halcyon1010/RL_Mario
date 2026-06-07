#!/usr/bin/env bash
set -euo pipefail

env_label="$1"
env_id="$2"

cd "$HOME/Mario/mario_project"
source "$HOME/Mario/rllte-venv/bin/activate"
mkdir -p long_logs videos /mnt/c/Users/250010056

master="long_logs/dual_${env_label}_1000ep_master.log"
: > "$master"
for tag in ppo_ngu_1m ppo_re3_1m ppo_e3b_1m; do
  : > "long_logs/${tag}_dual_${env_label}_1000ep.log"
done

printf 'TASK_DUAL_%s_1000_START %s env_id=%s\n' "$env_label" "$(date '+%Y-%m-%d %H:%M:%S %z')" "$env_id" | tee -a "$master"

run_one() {
  tag="$1"
  run_dir="$2"
  log="long_logs/${tag}_dual_${env_label}_1000ep.log"
  reward_out="videos/${tag}_dual_${env_label}_reward_best_1000ep.mp4"
  x_out="videos/${tag}_dual_${env_label}_x_best_1000ep.mp4"

  printf 'START %s %s\n' "$tag" "$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$master"
  python -u record_dual_best_checkpoint_videos.py \
    --config "${run_dir}/config.yaml" \
    --checkpoint "${run_dir}/checkpoints/best.pt" \
    --output-reward "$reward_out" \
    --output-x "$x_out" \
    --episodes 1000 \
    --mode sample \
    --env-id "$env_id" \
    --device cpu \
    --fps 30 > "$log" 2>&1

  cp -f "$reward_out" "/mnt/c/Users/250010056/${tag}_dual_${env_label}_reward_best_1000ep_final.mp4"
  cp -f "$x_out" "/mnt/c/Users/250010056/${tag}_dual_${env_label}_x_best_1000ep_final.mp4"
  tail -n 40 "$log" >> "$master"
  printf 'DONE %s %s\n' "$tag" "$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$master"
}

run_one ppo_ngu_1m runs/ppo_ngu_1m_seed1_1780568836 &
pid1=$!
sleep 2
run_one ppo_re3_1m runs/ppo_re3_1m_seed1_1780587129 &
pid2=$!
sleep 2
run_one ppo_e3b_1m runs/ppo_e3b_1m_seed1_1780604303 &
pid3=$!

status=0
while true; do
  live=0
  for pid in "$pid1" "$pid2" "$pid3"; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live + 1)); fi
  done
  printf 'HEARTBEAT live=%s time=%s\n' "$live" "$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$master"
  for tag in ppo_ngu_1m ppo_re3_1m ppo_e3b_1m; do
    line=$(grep 'episode_summary=' "long_logs/${tag}_dual_${env_label}_1000ep.log" 2>/dev/null | tail -n 1 || true)
    [ -n "$line" ] && printf '%s %s\n' "$tag" "$line" | tee -a "$master"
  done
  [ "$live" -eq 0 ] && break
  sleep 120
done

wait "$pid1" || status=1
wait "$pid2" || status=1
wait "$pid3" || status=1

if [ "$status" -eq 0 ]; then
  printf 'ALL_DONE %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$master"
else
  printf 'SOME_FAILED %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$master"
fi
exit "$status"
