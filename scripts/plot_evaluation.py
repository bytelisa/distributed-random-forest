import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import ConfusionMatrixDisplay

# Static PNG plots comparing the distributed system against the
# non-distributed baseline, across whichever tasks have been evaluated
# (scripts/evaluation.py -> accuracy_results_*.csv/evaluation_results_*.csv/
# predictions_*.csv, scripts/evaluate_oob.py -> oob_results_*.csv,
# scripts/validation_curve.py -> validation_curve_*.csv).


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


def load_predictions(input_dir: str) -> dict:
    """{task_name: {"true": [...], "distributed": [...], "baseline": [...]}}."""
    tasks = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "predictions_*.csv"))):
        with open(path, newline="") as f:
            reader = list(csv.DictReader(f))
        if reader:
            tasks[task_name(path, "predictions_")] = {
                "true": [r["true_label"] for r in reader],
                "distributed": [r["distributed_prediction"] for r in reader],
                "baseline": [r["baseline_prediction"] for r in reader],
            }
    return tasks


def plot_confusion_matrices(predictions_tasks: dict, output_dir: str, suffix: str = ""):
    if not predictions_tasks:
        print("[PlotEvaluation] no predictions_*.csv found, skipping confusion matrices")
        return

    for name, preds in predictions_tasks.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        ConfusionMatrixDisplay.from_predictions(preds["true"], preds["distributed"], ax=axes[0], colorbar=False)
        axes[0].set_title("Distributed")
        ConfusionMatrixDisplay.from_predictions(preds["true"], preds["baseline"], ax=axes[1], colorbar=False)
        axes[1].set_title("Baseline")
        fig.suptitle(f"Confusion matrix - {name}")
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"confusion_matrix_{name}{suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[PlotEvaluation] saved {out_path}")


def load_validation_curves(input_dir: str) -> dict:
    curves = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "validation_curve_*.csv"))):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            curves[task_name(path, "validation_curve_")] = sorted(rows, key=lambda r: int(r["n_estimators"]))
    return curves


def plot_validation_curves(curves: dict, output_dir: str, suffix: str = ""):
    if not curves:
        print("[PlotEvaluation] no validation_curve_*.csv found, skipping validation curve plots")
        return

    for name, rows in curves.items():
        n_values = [int(r["n_estimators"]) for r in rows]
        classification = rows[0]["oob_score_name"] == "oob_accuracy"
        oob_values = [float(r["oob_score"]) for r in rows]
        oob_curve = [1 - v for v in oob_values] if classification else oob_values
        train_times = [float(r["train_time_s"]) for r in rows]

        # Only the OOB curve is plotted: the held-out set used by
        # validation_curve.py is small enough that its error moves in
        # steps of one row, so it carries no information about convergence.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(n_values, oob_curve, marker="o")
        axes[0].set_xlabel("Number of trees")
        axes[0].set_ylabel("OOB error rate" if classification else "OOB RMSE")
        axes[0].set_title("Validation curve")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(n_values, train_times, marker="o", color="tab:red")
        axes[1].set_xlabel("Number of trees")
        axes[1].set_ylabel("Training time (s)")
        axes[1].set_title("Training time vs forest size")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"OOB convergence - {name}")
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"validation_curve_{name}{suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[PlotEvaluation] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot distributed-vs-baseline accuracy/timing/OOB/confusion-matrix/validation-curve comparisons.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluation = cfg["evaluation"]
    suffix = evaluation.get("plot_suffix", "")
    input_dir = args.input_dir or evaluation["output_dir"]
    output_dir = args.output_dir or os.path.join(input_dir, f"plots{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    accuracy_tasks = load_rows_by_model(input_dir, "accuracy_results_")
    timing_tasks = load_rows_by_model(input_dir, "evaluation_results_")
    oob_tasks = load_oob(input_dir)
    predictions_tasks = load_predictions(input_dir)
    validation_curves = load_validation_curves(input_dir)

    plot_accuracy_vs_oob(accuracy_tasks, oob_tasks, os.path.join(output_dir, f"accuracy_vs_oob{suffix}.png"))
    plot_time_comparison(timing_tasks, os.path.join(output_dir, f"time_comparison{suffix}.png"))
    plot_confusion_matrices(predictions_tasks, output_dir, suffix)
    plot_validation_curves(validation_curves, output_dir, suffix)


if __name__ == "__main__":
    main()
