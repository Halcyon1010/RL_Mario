$ErrorActionPreference = "Continue"

Write-Output "--- WSL-PROCESSES ---"
Get-Process wsl -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime | Format-Table -AutoSize

Write-Output "--- WSL-STATUS ---"
wsl -l -v

Write-Output "--- LINUX-STATUS ---"
$LinuxCommand = @'
set -euo pipefail
ps -eo pid,ppid,etime,stat,cmd | grep -E 'remote_run_ppo|experiments\.ppo|python -u' | grep -v grep || true
cd "$HOME/Mario/mario_project"
echo "--- master ---"
tail -n 30 long_logs/master.log 2>/dev/null || true
echo "--- latest ngu run ---"
run=$(ls -td runs/ppo_ngu_1m_* 2>/dev/null | head -1 || true)
echo "run=$run"
if [ -n "$run" ]; then
  echo "update_rows=$(($(wc -l < "$run/update_metrics.csv" 2>/dev/null || echo 1)-1))"
  tail -n 3 "$run/update_metrics.csv" 2>/dev/null || true
fi
echo "--- gpu ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
'@

wsl -d Ubuntu -- bash -lc $LinuxCommand
