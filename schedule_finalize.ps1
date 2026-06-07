$ErrorActionPreference = "Continue"

$TaskName = "MarioPPOIntrinsicFinalize"
$TaskRun = "cmd.exe /c C:\Users\250010056\run_wsl_finalize.cmd"

schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
schtasks /Create /TN $TaskName /SC ONCE /ST 23:59 /TR $TaskRun /F
schtasks /Run /TN $TaskName

Start-Sleep -Seconds 5
schtasks /Query /TN $TaskName /V /FO LIST
Get-Process cmd, wsl -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime | Format-Table -AutoSize
