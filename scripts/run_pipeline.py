import argparse
import json
import os
import subprocess
import sys

import yaml

sys.path.append(os.getcwd())

from scripts.evaluation import run_evaluation  # noqa: E402

# Runs the whole performance-analysis campaign unattended: scalability,
# per-task accuracy/OOB, the validation curve, then the plots. Every step
# already writes its results incrementally and skips what's already done, so
# interrupting and re-running this picks up where it left off.


def task_name(train_request_path):
    with open(train_request_path) as f:
        req = json.load(f)
    return os.path.splitext(os.path.basename(req["dataset_url"]))[0], req


def run_step(cmd):
    print(f"[Pipeline] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the full performance-analysis pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="re-run steps whose output already exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pipeline = cfg["pipeline"]
    python = sys.executable
    force_flag = ["--force"] if args.force else []

    if pipeline.get("run_scalability", True):
        run_step([python, "scripts/synthetic.py", "--config", args.config])
        run_step([python, "scripts/scalability.py", "--config", args.config] + force_flag)

    output_dir = cfg["evaluation"]["output_dir"]
    for request_path in pipeline["tasks"]:
        name, req = task_name(request_path)
        accuracy_csv = os.path.join(output_dir, f"accuracy_results_{name}.csv")
        if os.path.exists(accuracy_csv) and not args.force:
            print(f"[Pipeline] {name}: already evaluated ({accuracy_csv}), skipping")
            continue
        print(f"[Pipeline] evaluating {name}")
        run_evaluation(cfg, req, name)

    if pipeline.get("run_validation_curve", True):
        run_step([python, "scripts/validation_curve.py", "--config", args.config] + force_flag)

    run_step([python, "scripts/plot_scalability.py", "--config", args.config])
    run_step([python, "scripts/plot_evaluation.py", "--config", args.config])

    print("[Pipeline] done.")


if __name__ == "__main__":
    main()
