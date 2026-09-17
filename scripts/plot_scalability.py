import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

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


def plot_speedup(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    worker_counts = sorted(df["num_workers"].unique())
    ax.plot(worker_counts, worker_counts, linestyle="--", color="gray", label="ideal linear speedup")
    for size_label, group in df.groupby("dataset_size_n", sort=True):
        group = group.sort_values("num_workers")
        label = group["dataset_size"].iloc[0]
        single_worker_time = group.loc[group["num_workers"] == 1, "train_mean_s"].iloc[0]
        speedup = single_worker_time / group["train_mean_s"]
        ax.plot(group["num_workers"], speedup, marker="o", label=f"{label} rows")
    style_worker_axis(ax, worker_counts)
    ax.set_ylabel("Speedup (T(1 worker) / T(n workers))")
    ax.set_title("Training speedup vs worker count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot scripts/scalability.py's CSV output.")
    parser.add_argument("--csv", default="performance/scalability/scalability_results.csv")
    parser.add_argument("--output-dir", default="performance/scalability/plots")
    args = parser.parse_args()

    df = load(args.csv)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_time(df, "train_mean_s", "train_ci95_s", "Training time (s)",
              "Training time vs worker count (95% CI)",
              os.path.join(args.output_dir, "training_time.png"))
    plot_time(df, "predict_mean_s", "predict_ci95_s", "Prediction time (s)",
              "Prediction time vs worker count (95% CI)",
              os.path.join(args.output_dir, "predict_time.png"))
    plot_speedup(df, os.path.join(args.output_dir, "speedup.png"))


if __name__ == "__main__":
    main()
