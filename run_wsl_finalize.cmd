@echo off
echo FINALIZE_START %DATE% %TIME% > C:\Users\250010056\wsl_finalize.log
wsl -d Ubuntu -- bash -lc "cd \"$HOME/Mario/mario_project\" && cp /mnt/c/Users/250010056/generate_run_report_ppo_intrinsic.py ./generate_run_report_ppo_intrinsic.py && while pgrep -f '[r]emote_run_ppo_intrinsic_1m.sh' > /dev/null || pgrep -f '[e]xperiments.ppo_' > /dev/null || pgrep -f '[m]ario_rl.evaluate_checkpoint' > /dev/null; do sleep 300; done; source \"$HOME/Mario/rllte-venv/bin/activate\" && python generate_run_report_ppo_intrinsic.py && tar -czf /mnt/c/Users/250010056/mario_ppo_intrinsic_artifacts.tar.gz RUN_REPORT_ppo_ngu_re3_e3b.md videos long_logs" >> C:\Users\250010056\wsl_finalize.log 2>&1
echo FINALIZE_EXIT %ERRORLEVEL% %DATE% %TIME% >> C:\Users\250010056\wsl_finalize.log
