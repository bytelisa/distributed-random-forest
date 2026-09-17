import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt

# Static PNG plots comparing the distributed system against the
# non-distributed baseline, across whichever tasks have been evaluated
# (scripts/evaluation.py -> accuracy_results_*.csv/evaluation_results_*.csv,
# scripts/evaluate_oob.py -> oob_results_*.csv, all in performance/evaluation/).


def task_name(path: str, prefix: str) -> str:
    return os.path.basename(path)[len(prefix):-len(".csv")]


def load_rows_by_model(input_dir: str, prefix: str) -> dict:
    """{task_name: {"distributed": row, "baseline": row}}, skipping any file missing either model."""
    tasks = {}
    for path in sorted(glob.glob(os.path.join(input_dir, f"{prefix}*.csv"))):
        with open(path, newline="") as f:
            rows = {row["model"]: row for row in csv.DictReader(f)}
        if "distributed" in rows and "baseline" in rows:
            tasks[task_name(path, prefix)] = rows
    return tasks


def load_oob(input_dir: str) -> dict:
    """{task_name: row} - one OOB score per model, no baseline counterpart."""
    oob = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "oob_results_*.csv"))):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            oob[task_name(path, "oob_results_")] = rows[0]
    return oob


def plot_accuracy_vs_oob(accuracy_tasks: dict, oob_tasks: dict, out_path: str):
    if not accuracy_tasks:
        print("[PlotEvaluation] no accuracy_results_*.csv found, skipping accuracy/OOB plot")
        return

    groups = [("classification", "accuracy", "oob_accuracy", "Accuracy"),
              ("regression", "rmse", "oob_rmse", "RMSE")]
    present = [g for g in groups if any(rows["distributed"]["task_type"] == g[0] for rows in accuracy_tasks.values())]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False)
    for ax, (task_type, metric_key, oob_key, ylabel) in zip(axes[0], present):
        names = [n for n, rows in accuracy_tasks.items() if rows["distributed"]["task_type"] == task_type]
        x = range(len(names))
        width = 0.25
        dist_vals = [float(accuracy_tasks[n]["distributed"][metric_key]) for n in names]
        base_vals = [float(accuracy_tasks[n]["baseline"][metric_key]) for n in names]

        ax.bar([i - width for i in x], dist_vals, width, label="distributed (test set)")
        ax.bar(list(x), base_vals, width, label="baseline (test set)")

        oob_x = [i + width for i, n in zip(x, names) if n in oob_tasks]
        oob_y = [float(oob_tasks[n][oob_key]) for n in names if n in oob_tasks]
        if oob_y:
            ax.bar(oob_x, oob_y, width, label="distributed (out-of-bag)", color="tab:green")

        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel)
        ax.set_title(task_type.capitalize())
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Distributed vs baseline vs out-of-bag")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PlotEvaluation] saved {out_path}")


def plot_time_comparison(timing_tasks: dict, out_path: str):
    if not timing_tasks:
        print("[PlotEvaluation] no evaluation_results_*.csv found, skipping time comparison plot")
        return

    names = list(timing_tasks.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (mean_col, ylabel) in zip(axes, [("train_mean_s", "Training time (s)"), ("predict_mean_s", "Prediction time (s)")]):
        std_col = mean_col.replace("mean", "std")
        x = range(len(names))
        width = 0.35
        dist_vals = [float(timing_tasks[n]["distributed"][mean_col]) for n in names]
        base_vals = [float(timing_tasks[n]["baseline"][mean_col]) for n in names]
        dist_err = [float(timing_tasks[n]["distributed"][std_col]) for n in names]
        base_err = [float(timing_tasks[n]["baseline"][std_col]) for n in names]

        ax.bar([i - width / 2 for i in x], dist_vals, width, yerr=dist_err, capsize=3, label="distributed")
        ax.bar([i + width / 2 for i in x], base_vals, width, yerr=base_err, capsize=3, label="baseline")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_ylabel(f"{ylabel} (log scale)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Distributed vs baseline: training/prediction time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PlotEvaluation] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot distributed-vs-baseline accuracy/timing/OOB comparisons.")
    parser.add_argument("--input-dir", default="performance/evaluation")
    parser.add_argument("--output-dir", default="performance/evaluation/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    accuracy_tasks = load_rows_by_model(args.input_dir, "accuracy_results_")
    timing_tasks = load_rows_by_model(args.input_dir, "evaluation_results_")
    oob_tasks = load_oob(args.input_dir)

    plot_accuracy_vs_oob(accuracy_tasks, oob_tasks, os.path.join(args.output_dir, "accuracy_vs_oob.png"))
    plot_time_comparison(timing_tasks, os.path.join(args.output_dir, "time_comparison.png"))


if __name__ == "__main__":
    main()
