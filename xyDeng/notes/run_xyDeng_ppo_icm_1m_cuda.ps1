$env:PYTHONPATH='C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario;C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario\rllte'
Set-Location 'C:\Users\250010109\data\RL_projects\real_mario_project\RL_Mario'
& 'C:\Users\250010109\AppData\Local\miniconda3\envs\mario\python.exe' -m experiments.ppo_icm --total-timesteps 1000000 --eval-interval 50000 --eval-episodes 5 --test-episodes 5 --save-interval 100000 --device cuda --experiment-name xyDeng_ppo_icm_1m_cuda --run-dir 'C:\Users\250010109\data\RL_projects\RL_Mario\xyDeng\runs' --no-wandb
