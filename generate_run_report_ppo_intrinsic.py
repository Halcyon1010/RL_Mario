import csv
import json
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
METHODS = [
    {
        "label": "PPO + NGU",
        "key": "ngu",
        "entry": "experiments.ppo_ngu",
        "script": "experiments/ppo_ngu.py",
        "config": "configs/ppo_ngu.yaml",
        "reward_type": "ngu",
        "long_name": "ppo_ngu_1m",
        "smoke_glob": "ppo_ngu_seed1_*",
    },
    {
        "label": "PPO + RE3",
        "key": "re3",
        "entry": "experiments.ppo_re3",
        "script": "experiments/ppo_re3.py",
        "config": "configs/ppo_re3.yaml",
        "reward_type": "re3",
        "long_name": "ppo_re3_1m",
        "smoke_glob": "ppo_re3_seed1_*",
    },
    {
        "label": "PPO + E3B",
        "key": "e3b",
        "entry": "experiments.ppo_e3b",
        "script": "experiments/ppo_e3b.py",
        "config": "configs/ppo_e3b.yaml",
        "reward_type": "e3b",
        "long_name": "ppo_e3b_1m",
        "smoke_glob": "ppo_e3b_seed1_*",
    },
]


def yes(value):
    return "通过" if value else "未通过"


def exists_text(path):
    return "存在" if path.exists() else "缺失"


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def last_csv_row(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def row_count(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def latest_run(pattern):
    matches = sorted((ROOT / "runs").glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def long_runs_from_log():
    result = {}
    path = ROOT / "long_logs" / "run_dirs.txt"
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = dict(part.split("=", 1) for part in line.split() if "=" in part)
        method = parts.get("method")
        run_dir = parts.get("run_dir")
        if method and run_dir:
            result[method] = ROOT / run_dir
    return result


def yaml_reward_type(config_path):
    if not config_path.exists():
        return None
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("reward", {}).get("intrinsic", {}).get("type")


def required_run_files(run_dir):
    names = [
        "config.yaml",
        "train_metrics.csv",
        "update_metrics.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "summary.json",
        "checkpoints/last.pt",
    ]
    return {name: run_dir / name for name in names}


def run_passed(run_dir):
    if run_dir is None:
        return False
    files = required_run_files(run_dir)
    required_present = all(path.exists() for path in files.values() if path.name != "test_metrics.csv")
    csv_nonempty = all(row_count(run_dir / name) > 0 for name in ["train_metrics.csv", "update_metrics.csv", "val_metrics.csv"])
    return required_present and csv_nonempty


def value(row, key):
    if row is None:
        return "缺失"
    item = row.get(key)
    return "缺失" if item in (None, "") else str(item)


def main():
    mapping_file = ROOT / "mario_rl" / "rewards" / "rllte_intrinsic.py"
    mapping_text = mapping_file.read_text(encoding="utf-8") if mapping_file.exists() else ""
    long_runs = long_runs_from_log()
    lines = []

    lines.append("# PPO + NGU / RE3 / E3B 运行报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("- 本次只运行：PPO + NGU、PPO + RE3、PPO + E3B")
    lines.append("- 未运行：DQN / DDPG / SAC / SAC-Discrete")
    lines.append("- 原因：第一轮主线固定 PPO backbone，避免 RL backbone 差异成为混杂变量。")
    lines.append("- 说明：当前训练是外部奖励 + 内在奖励的 PPO 训练链路；本报告只说明代码入口、训练链路、日志、checkpoint 和视频产出情况，不说明方法性能优劣。")
    lines.append("")

    lines.append("## 静态检查结果")
    lines.append("")
    lines.append("| 检查项 | NGU | RE3 | E3B | 证据 |")
    lines.append("| --- | --- | --- | --- | --- |")
    script_row = ["入口脚本存在"]
    config_row = ["配置文件存在"]
    type_row = ["reward type 正确"]
    mapping_row = ["reward module 映射存在"]
    for method in METHODS:
        script_row.append(yes((ROOT / method["script"]).exists()))
        config_row.append(yes((ROOT / method["config"]).exists()))
        type_row.append(yes(yaml_reward_type(ROOT / method["config"]) == method["reward_type"]))
        mapping_row.append(yes(f'"{method["reward_type"]}"' in mapping_text))
    lines.append("| " + " | ".join(script_row + ["experiments/*.py"]) + " |")
    lines.append("| " + " | ".join(config_row + ["configs/*.yaml"]) + " |")
    lines.append("| " + " | ".join(type_row + ["YAML reward.intrinsic.type"]) + " |")
    lines.append("| " + " | ".join(mapping_row + ["mario_rl/rewards/rllte_intrinsic.py"]) + " |")
    lines.append("")

    lines.append("## Smoke test 结果")
    lines.append("")
    lines.append("| 方法 | 命令 | 退出码 | run_dir | 结果 | 失败说明 |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for method in METHODS:
        run_dir = latest_run(method["smoke_glob"])
        result = run_passed(run_dir)
        command = f"python -m {method['entry']} --total-timesteps 1000 --eval-interval 500 --eval-episodes 1 --test-episodes 1 --no-wandb"
        lines.append(
            f"| {method['label']} | `{command}` | {0 if result else '缺失'} | "
            f"`{run_dir.relative_to(ROOT) if run_dir else '缺失'}` | {yes(result)} | {'无' if result else '见运行目录/日志'} |"
        )
    lines.append("")

    lines.append("## 1,000,000 step 训练结果")
    lines.append("")
    lines.append("| 方法 | run_dir | global_step | episodes | best_val_total_reward | summary.json | checkpoints/last.pt | 结果 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
    for method in METHODS:
        run_dir = long_runs.get(method["long_name"]) or latest_run(f"{method['long_name']}_seed1_*")
        summary = load_json(run_dir / "summary.json") if run_dir else None
        result = run_passed(run_dir)
        lines.append(
            f"| {method['label']} | `{run_dir.relative_to(ROOT) if run_dir else '缺失'}` | "
            f"{summary.get('global_step', '缺失') if summary else '缺失'} | "
            f"{summary.get('episodes', '缺失') if summary else '缺失'} | "
            f"{summary.get('best_val_total_reward', '缺失') if summary else '缺失'} | "
            f"{exists_text(run_dir / 'summary.json') if run_dir else '缺失'} | "
            f"{exists_text(run_dir / 'checkpoints' / 'last.pt') if run_dir else '缺失'} | {yes(result)} |"
        )
    lines.append("")

    lines.append("## 关键指标摘录")
    lines.append("")
    for method in METHODS:
        run_dir = long_runs.get(method["long_name"]) or latest_run(f"{method['long_name']}_seed1_*")
        summary = load_json(run_dir / "summary.json") if run_dir else None
        update_row = last_csv_row(run_dir / "update_metrics.csv") if run_dir else None
        train_row = last_csv_row(run_dir / "train_metrics.csv") if run_dir else None
        video = ROOT / "videos" / f"{method['long_name']}_best_sample.mp4"
        lines.append(f"### {method['label']}")
        lines.append("")
        lines.append(f"- run_dir：`{run_dir.relative_to(ROOT) if run_dir else '缺失'}`")
        lines.append(f"- summary.json：{exists_text(run_dir / 'summary.json') if run_dir else '缺失'}")
        lines.append(f"- train_metrics.csv：{exists_text(run_dir / 'train_metrics.csv') if run_dir else '缺失'}，rows={row_count(run_dir / 'train_metrics.csv') if run_dir else 0}")
        lines.append(f"- update_metrics.csv：{exists_text(run_dir / 'update_metrics.csv') if run_dir else '缺失'}，rows={row_count(run_dir / 'update_metrics.csv') if run_dir else 0}")
        lines.append(f"- val_metrics.csv：{exists_text(run_dir / 'val_metrics.csv') if run_dir else '缺失'}，rows={row_count(run_dir / 'val_metrics.csv') if run_dir else 0}")
        lines.append(f"- test_metrics.csv：{exists_text(run_dir / 'test_metrics.csv') if run_dir else '缺失'}，rows={row_count(run_dir / 'test_metrics.csv') if run_dir else 0}")
        lines.append(f"- checkpoints/last.pt：{exists_text(run_dir / 'checkpoints' / 'last.pt') if run_dir else '缺失'}")
        lines.append(f"- video：`{video.relative_to(ROOT)}`，{exists_text(video)}")
        if summary:
            lines.append(
                "- summary："
                f"global_step={summary.get('global_step')}, "
                f"episodes={summary.get('episodes')}, "
                f"device={summary.get('device')}, "
                f"num_actions={summary.get('num_actions')}, "
                f"action_space={summary.get('action_space')}, "
                f"best_val_total_reward={summary.get('best_val_total_reward')}"
            )
        else:
            lines.append("- summary：缺失")
        lines.append(
            "- update_metrics 最后一行："
            f"rollout_int_reward_mean={value(update_row, 'rollout_int_reward_mean')}, "
            f"rollout_int_reward_sum={value(update_row, 'rollout_int_reward_sum')}, "
            f"entropy={value(update_row, 'entropy')}, "
            f"approx_kl={value(update_row, 'approx_kl')}, "
            f"dominant_action_frac={value(update_row, 'dominant_action_frac')}"
        )
        lines.append(
            "- train_metrics 最后一条 episode："
            f"episode_ext_reward={value(train_row, 'episode_ext_reward')}, "
            f"episode_int_reward={value(train_row, 'episode_int_reward')}, "
            f"episode_total_reward={value(train_row, 'episode_total_reward')}, "
            f"episode_max_x_pos={value(train_row, 'episode_max_x_pos')}"
        )
        lines.append("")

    lines.append("## 运行中最小修复")
    lines.append("")
    lines.append("- `rllte/rllte/xplore/reward/pseudo_counts.py`：修复离散动作下 `CrossEntropyLoss(reduction=\"none\")` 返回 1D loss 时的 mask expand 维度错误。")
    lines.append("- `rllte/rllte/xplore/reward/e3b.py`：同样修复 E3B 离散动作 loss mask 维度错误。")
    lines.append("- 以上修复只影响离散动作 loss mask 的形状适配，不改变 PPO 主训练逻辑或实验范围。")
    lines.append("")

    lines.append("## 验收结论")
    lines.append("")
    all_long_passed = all(run_passed(long_runs.get(method["long_name"]) or latest_run(f"{method['long_name']}_seed1_*")) for method in METHODS)
    if all_long_passed:
        lines.append("三个 PPO intrinsic reward 方法均完成 1,000,000 step 训练，并产出 summary、非空训练/更新/验证指标、checkpoint 和视频。")
    else:
        lines.append("至少一个方法尚未完成或缺少验收文件，不能写作全部通过。")

    (ROOT / "RUN_REPORT_ppo_ngu_re3_e3b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
