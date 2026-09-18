import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import yaml

# Static PNG plots from scripts/scalability.py's CSV output, for the report.


def parse_size(label: str) -> int:
    return int(label[:-1]) * 1000 if label.endswith("k") else int(label)


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["dataset_size_n"] = df["dataset_size"].apply(parse_size)
    return df.sort_values(["dataset_size_n", "num_workers"])


def style_worker_axis(ax, worker_counts):
    ax.set_xlabel("Worker count")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(worker_counts))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)


def plot_time(df: pd.DataFrame, column: str, ci_column: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for size_label, group in df.groupby("dataset_size_n", sort=True):
        label = group["dataset_size"].iloc[0]
        ax.errorbar(group["num_workers"], group[column], yerr=group[ci_column],
                     marker="o", capsize=3, label=f"{label} rows")
    style_worker_axis(ax, df["num_workers"].unique())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved {out_path}")


def speedup(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("num_workers")
    single_worker_time = group.loc[group["num_workers"] == 1, "train_mean_s"].iloc[0]
    return single_worker_time / group["train_mean_s"]


def plot_speedup(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    worker_counts = sorted(df["num_workers"].unique())
    ax.plot(worker_counts, worker_counts, linestyle="--", color="gray", label="ideal linear speedup")
    for size_label, group in df.groupby("dataset_size_n", sort=True):
        group = group.sort_values("num_workers")
        label = group["dataset_size"].iloc[0]
        ax.plot(group["num_workers"], speedup(group), marker="o", label=f"{label} rows")
    style_worker_axis(ax, worker_counts)
    ax.set_ylabel("Speedup (T(1 worker) / T(n workers))")
    ax.set_title("Training speedup vs worker count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved {out_path}")


def plot_efficiency(df: pd.DataFrame, out_path: str):
    """Parallel efficiency = speedup(n) / n - how much of each added worker's
    theoretical capacity is actually recovered, isolating the effect of fixed
    per-worker overhead (gRPC round trips, dataset download, tree uploads)
    that the speedup plot shows but doesn't quantify directly."""
    fig, ax = plt.subplots(figsize=(7, 5))
    worker_counts = sorted(df["num_workers"].unique())
    ax.axhline(1.0, linestyle="--", color="gray", label="ideal linear speedup")
    for size_label, group in df.groupby("dataset_size_n", sort=True):
        group = group.sort_values("num_workers")
        label = group["dataset_size"].iloc[0]
        efficiency = speedup(group) / group["num_workers"]
        ax.plot(group["num_workers"], efficiency, marker="o", label=f"{label} rows")
    style_worker_axis(ax, worker_counts)
    ax.set_ylabel("Parallel efficiency (speedup / worker count)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Parallel efficiency vs worker count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved {out_path}")


def plot_boxplot(raw_csv: str, out_path: str):
    if not os.path.exists(raw_csv):
        print(f"[Plot] {raw_csv} not found, skipping box plot")
        return

    raw = pd.read_csv(raw_csv)
    raw["dataset_size_n"] = raw["dataset_size"].apply(parse_size)
    sizes = sorted(raw["dataset_size_n"].unique())

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 5), squeeze=False, sharey=True)
    for ax, size_n in zip(axes[0], sizes):
        group = raw[raw["dataset_size_n"] == size_n]
        worker_counts = sorted(group["num_workers"].unique())
        data = [group[group["num_workers"] == w]["train_s"] for w in worker_counts]
        ax.boxplot(data, tick_labels=worker_counts)
        label = group["dataset_size"].iloc[0]
        ax.set_title(f"{label} rows")
        ax.set_xlabel("Worker count")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel("Training time (s)")

    fig.suptitle("Training time distribution vs worker count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot scripts/scalability.py's CSV output.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    scal = cfg["scalability"]
    scal_output_dir = scal["output_dir"]
    suffix = scal.get("plot_suffix", "")
    csv_path = args.csv or os.path.join(scal_output_dir, "scalability_results.csv")
    raw_csv = os.path.join(scal_output_dir, "scalability_runs.csv")
    output_dir = args.output_dir or os.path.join(scal_output_dir, f"plots{suffix}")

    df = load(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    plot_time(df, "train_mean_s", "train_ci95_s", "Training time (s)",
              "Training time vs worker count (95% CI)",
              os.path.join(output_dir, f"training_time{suffix}.png"))
    plot_time(df, "predict_mean_s", "predict_ci95_s", "Prediction time (s)",
              "Prediction time vs worker count (95% CI)",
              os.path.join(output_dir, f"predict_time{suffix}.png"))
    plot_speedup(df, os.path.join(output_dir, f"speedup{suffix}.png"))
    plot_efficiency(df, os.path.join(output_dir, f"efficiency{suffix}.png"))
    plot_boxplot(raw_csv, os.path.join(output_dir, f"training_time_boxplot{suffix}.png"))


if __name__ == "__main__":
    main()
