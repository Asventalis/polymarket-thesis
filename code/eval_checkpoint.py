"""eval_checkpoint.py — evaluate trained checkpoints on other days with train_full.py --init-from.

For every label the flags of the original run are read back from its run_config_<label>.json, so that the
model is rebuilt exactly as it was trained (encoder, window, memory, features, trees, ...); the training
flags are replaced by an evaluation schedule: no training days, --warm-days streamed without learning to
rebuild the node states and the LOB buffers, then --val-days recorded.  Runs are sequential (one process
per GPU) and idempotent: a label whose predictions already exist in --out-dir is skipped, so a queue can be
relaunched.

    # consistency check of the mechanism on the validation day (no test data involved)
    ./run_dev.sh eval_checkpoint.py --device cuda:1 --warm-days 11-12 --val-days 13 --suffix _recheck --out-dir recheck \\
        --labels 1h_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0
    # the test set (after the author's go), with the feature directories that cover the test days
    ./run_dev.sh eval_checkpoint.py --device cuda:1 --warm-days 22-23 --val-days 24-26 --suffix _test --out-dir test_runs \\
        --set hand_dir=/data/v.pampinella/agy_project/data/derived/hand_features_test \\
        --set tree_dir=/data/v.pampinella/agy_project/data/derived/tree_logprobs_1h_test \\
        --labels LABEL1 LABEL2 ...
    --dry-run prints the commands without running them.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# flags of the training schedule and of this evaluation, never copied from the original run
DROPPED = {"train_days", "val_days", "warm_days", "device", "out_dir", "run_label", "init_from", "ema", "epochs",
           "max_hours", "max_blocks_per_hour", "save_val_predictions", "started_at"}


def build_command(config: dict, label: str, ckpt: Path, args, overrides: dict) -> list:
    cmd = ([sys.executable] if not args.runner else []) + ["train_full.py"]
    for key, value in config.items():
        if key in DROPPED or key in args.drop:
            continue
        if key in overrides:
            value = overrides[key]
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            cmd.append(flag)
        else:
            cmd += [flag, str(value)]
    for key, value in overrides.items():          # overrides of flags absent from an old config
        if key not in config and key not in DROPPED:
            cmd += ["--" + key.replace("_", "-"), str(value)]
    cmd += ["--init-from", str(ckpt), "--train-days", "", "--warm-days", args.warm_days, "--val-days", args.val_days,
            "--device", args.device, "--out-dir", str(args.out_dir), "--run-label", label + args.suffix,
            "--ema", "0", "--epochs", "1", "--save-val-predictions"]
    if args.max_hours:
        cmd += ["--max-hours", str(args.max_hours)]
    if args.max_blocks_per_hour:
        cmd += ["--max-blocks-per-hour", str(args.max_blocks_per_hour)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="labels of the original runs (model_<label>_final.pt)")
    ap.add_argument("--src-dir", type=Path, default=Path("/data/v.pampinella/agy_project_orient"),
                    help="directory of the checkpoints and run_config files")
    ap.add_argument("--device", required=True)
    ap.add_argument("--warm-days", required=True, help="days streamed without learning before the evaluated days")
    ap.add_argument("--val-days", required=True, help="evaluated days")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suffix", default="_test", help="appended to the label of the evaluation run")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a flag of the original run (e.g. hand_dir=/path, tree_dir=/path, tsfm_dir=/path)")
    ap.add_argument("--drop", action="append", default=[], metavar="KEY", help="ignore a key of the original config")
    ap.add_argument("--max-hours", type=int, default=None, help="smoke test")
    ap.add_argument("--max-blocks-per-hour", type=int, default=None, help="smoke test")
    ap.add_argument("--runner", default="", help="prefix that starts train_full.py (e.g. ./run_dev.sh when this script runs "
                                                 "outside the container); default: this interpreter, i.e. run this script "
                                                 "itself through ./run_dev.sh")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    overrides = dict(kv.split("=", 1) for kv in args.set)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for label in args.labels:
        cfg_path = args.src_dir / f"run_config_{label}.json"
        ckpt = args.src_dir / f"model_{label}_final.pt"
        for p in (cfg_path, ckpt):
            if not p.is_file():
                raise SystemExit(f"{p} not found")
        done = args.out_dir / f"val_predictions_{label}{args.suffix}.parquet"
        if done.exists():
            print(f">>> {label}{args.suffix}: predictions exist, skipped", flush=True)
            continue
        config = json.loads(cfg_path.read_text())
        cmd = build_command(config, label, ckpt, args, overrides)
        if args.runner:
            cmd = args.runner.split() + cmd
        log = args.out_dir / f"{label}{args.suffix}.log"
        print(f">>> {label}{args.suffix} on {args.device}  ({datetime.now():%Y-%m-%d %H:%M:%S})\n    " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        with open(log, "a") as f:
            f.write("# " + " ".join(cmd) + "\n")
            f.flush()
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        status = "DONE" if rc == 0 and done.exists() else f"FAILED (exit {rc})"
        print(f">>> {label}{args.suffix} {status} in {(time.time() - t0) / 3600:.2f} h  ({datetime.now():%Y-%m-%d %H:%M:%S})", flush=True)
    print(">>> ALL DONE", flush=True)


if __name__ == "__main__":
    main()
