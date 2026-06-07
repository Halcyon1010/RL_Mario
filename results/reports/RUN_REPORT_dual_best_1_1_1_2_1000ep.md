# Mario PPO Intrinsic Dual-Best 1000 Episode Report

Generated: 2026-06-06 16:11 +0800

This report summarizes the same-batch 1000 sampled episode evaluation for three 1M-step PPO intrinsic agents. Each run saved two videos from the same 1000 episodes:

- `reward_best`: episode with the highest total external reward.
- `x_best`: episode with the farthest `max_x_pos`.

## SuperMarioBros-1-1-v0

| Method | Reward-best episode | Best total reward | Reward-best max_x_pos | X-best episode | Best max_x_pos | X-best total reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO-NGU | 568 | 3014.00 | 3161 | 294 | 3161 | 3009.00 |
| PPO-RE3 | 21 | 3017.00 | 3161 | 21 | 3161 | 3017.00 |
| PPO-E3B | 567 | 2344.00 | 2472 | 451 | 2475 | 2343.00 |

## SuperMarioBros-1-2-v0

| Method | Reward-best episode | Best total reward | Reward-best max_x_pos | X-best episode | Best max_x_pos | X-best total reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO-NGU | 971 | 2070.00 | 2272 | 971 | 2272 | 2070.00 |
| PPO-RE3 | 261 | 1747.00 | 1940 | 261 | 1940 | 1747.00 |
| PPO-E3B | 617 | 1810.00 | 1955 | 617 | 1955 | 1810.00 |

## Local Videos

All 12 final videos were copied to `/Users/chengzhiwei/Documents/Mario/outputs/`.

### 1-1

- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_ngu_1m_dual_1_1_reward_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_ngu_1m_dual_1_1_x_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_re3_1m_dual_1_1_reward_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_re3_1m_dual_1_1_x_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_e3b_1m_dual_1_1_reward_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_e3b_1m_dual_1_1_x_best_1000ep_final.mp4`

### 1-2

- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_ngu_1m_dual_1_2_reward_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_ngu_1m_dual_1_2_x_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_re3_1m_dual_1_2_reward_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_re3_1m_dual_1_2_x_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_e3b_1m_dual_1_2_reward_best_1000ep_final.mp4`
- `/Users/chengzhiwei/Documents/Mario/outputs/ppo_e3b_1m_dual_1_2_x_best_1000ep_final.mp4`

## Notes

- For 1-1, NGU and RE3 reached the known endpoint position `max_x_pos=3161` in this 1000 episode sample. E3B did not reach 3161 in this dual-best batch.
- For 1-2, none of the three transferred agents reached a full level endpoint in this sample. NGU achieved the best transfer distance and reward among the three.
- In several cases, reward-best and x-best are the same episode. Where they differ, both videos are preserved.
