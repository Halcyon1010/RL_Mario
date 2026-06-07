$ErrorActionPreference = "Stop"

$Process = Start-Process `
  -FilePath "cmd.exe" `
  -ArgumentList @("/c", "C:\Users\250010056\run_wsl_training.cmd") `
  -WindowStyle Hidden `
  -PassThru

Start-Sleep -Seconds 3
Write-Output "launcher_pid=$($Process.Id)"
Get-Process cmd, wsl -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime
