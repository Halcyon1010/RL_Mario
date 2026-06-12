import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_SPECS = [
    ("train_metrics.csv", "global_decision_step", "episode_ext_reward", "train_ext_reward.png", "Training Extrinsic Reward"),
    ("train_metrics.csv", "global_decision_step", "episode_int_reward", "train_int_reward.png", "Training Intrinsic Reward"),
    ("train_metrics.csv", "global_decision_step", "episode_total_reward", "train_total_reward.png", "Training Total Reward"),
    ("train_metrics.csv", "global_decision_step", "episode_max_x_pos", "train_max_x_pos.png", "Training Max X-position"),
    ("train_metrics.csv", "global_decision_step", "episode_mean_x_pos", "train_mean_x_pos.png", "Training Mean X-position"),
    ("train_metrics.csv", "global_decision_step", "episode_decision_steps", "train_episode_steps.png", "Training Episode Decision Steps"),
    ("train_metrics.csv", "global_decision_step", "visited_bins", "train_visited_bins.png", "Training Visited X-position Bins"),
    ("val_metrics.csv", "global_decision_step", "ext_reward", "val_ext_reward.png", "Validation Extrinsic Reward"),
    ("val_metrics.csv", "global_decision_step", "int_reward", "val_int_reward.png", "Validation Intrinsic Reward"),
    ("val_metrics.csv", "global_decision_step", "total_reward", "val_total_reward.png", "Validation Total Reward"),
    ("val_metrics.csv", "global_decision_step", "max_x_pos", "val_max_x_pos.png", "Validation Max X-position"),
    ("val_metrics.csv", "global_decision_step", "mean_x_pos", "val_mean_x_pos.png", "Validation Mean X-position"),
    ("val_metrics.csv", "global_decision_step", "decision_steps", "val_episode_steps.png", "Validation Episode Decision Steps"),
    ("test_metrics.csv", "global_decision_step", "total_reward", "test_total_reward.png", "Test Total Reward When Best Validation Improves"),
    ("test_metrics.csv", "global_decision_step", "max_x_pos", "test_max_x_pos.png", "Test Max X-position When Best Validation Improves"),
    ("update_metrics.csv", "global_decision_step", "policy_loss", "policy_loss.png", "PPO Policy Loss"),
    ("update_metrics.csv", "global_decision_step", "value_loss", "value_loss.png", "PPO Value Loss"),
    ("update_metrics.csv", "global_decision_step", "entropy", "entropy.png", "PPO Entropy"),
    ("update_metrics.csv", "global_decision_step", "approx_kl", "approx_kl.png", "PPO Approx KL"),
    ("update_metrics.csv", "global_decision_step", "dominant_action_frac", "dominant_action_frac.png", "Dominant Action Fraction"),
]


def run_label(run_dir: Path):
    name = run_dir.name
    parts = name.split("_seed")
    return parts[0] if parts else name


def read_metric(run_dir: Path, filename: str, x_col: str, y_col: str):
    path = run_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if x_col not in df.columns or y_col not in df.columns or df.empty:
        return None
    return df[[x_col, y_col]].dropna()


def smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(run_dirs, filename, x_col, y_col, output_path, title, smooth_window):
    plotted = False
    plt.figure(figsize=(9, 5))

    for run_dir in run_dirs:
        df = read_metric(run_dir, filename, x_col, y_col)
        if df is None:
            continue
        y = smooth(df[y_col], smooth_window)
        plt.plot(df[x_col], y, label=run_label(run_dir), linewidth=1.8)
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def expand_run_dirs(paths):
    run_dirs = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir() and (path / "train_metrics.csv").exists():
            run_dirs.append(path)
        elif path.is_dir():
            run_dirs.extend(sorted(child for child in path.iterdir() if child.is_dir() and (child / "train_metrics.csv").exists()))
    return run_dirs


def main():
    parser = argparse.ArgumentParser(description="Plot Mario RL experiment metrics.")
    parser.add_argument("paths", nargs="+", help="Run directories or a parent runs directory.")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--smooth", type=int, default=5, help="Moving-average smoothing window.")
    args = parser.parse_args()

    run_dirs = expand_run_dirs(args.paths)
    if not run_dirs:
        raise FileNotFoundError("No run directories with train_metrics.csv were found.")

    output_dir = Path(args.output_dir)
    print("plot=start")
    print("runs=" + ", ".join(str(path) for path in run_dirs))

    count = 0
    for filename, x_col, y_col, image_name, title in PLOT_SPECS:
        ok = plot_metric(
            run_dirs=run_dirs,
            filename=filename,
            x_col=x_col,
            y_col=y_col,
            output_path=output_dir / image_name,
            title=title,
            smooth_window=args.smooth,
        )
        if ok:
            count += 1
            print(f"plot=ok file={output_dir / image_name}")

    print(f"plot=done count={count}")


if __name__ == "__main__":
    main()
