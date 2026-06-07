@echo off
echo START %DATE% %TIME% > C:\Users\250010056\wsl_training_launcher.log
wsl -d Ubuntu -- bash -lc "cd \"$HOME/Mario/mario_project\" && cp /mnt/c/Users/250010056/remote_run_ppo_intrinsic_1m.sh ./remote_run_ppo_intrinsic_1m.sh && chmod +x remote_run_ppo_intrinsic_1m.sh && bash remote_run_ppo_intrinsic_1m.sh" >> C:\Users\250010056\wsl_training_launcher.log 2>&1
echo EXIT %ERRORLEVEL% %DATE% %TIME% >> C:\Users\250010056\wsl_training_launcher.log
