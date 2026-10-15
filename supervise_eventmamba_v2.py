"""Launch and supervise one V2 run; no automatic restart or parameter changes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--devices", default="1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.devices, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    devices = args.devices.split(",")
    prefix = [sys.executable, "-u"] if len(devices) == 1 else [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(devices)}"]
    command = prefix + [str(root / "train_eventmamba_v2_only.py"), "--output-dir", str(args.output_dir),
                        "--cache-dir", str(args.cache_dir), "--epochs", str(args.epochs),
                        "--batch-size", "32", "--num-workers", "4", "--prepare-workers", "4"]
    if args.resume:
        command += ["--resume", str(args.resume)]
    with (args.output_dir / "console.log").open("a") as console:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=console, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        pid_record = {"supervisor_pid": os.getpid(), "training_pid": child.pid, "command": command}
        (args.output_dir / "process.json").write_text(json.dumps(pid_record, indent=2) + "\n")
        print(json.dumps({"event": "launched", **pid_record}), flush=True)
        while True:
            try:
                gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
            except (OSError, subprocess.SubprocessError) as exc:
                gpu = str(exc)
            metrics = args.output_dir / "metrics.jsonl"
            age = time.time() - metrics.stat().st_mtime if metrics.exists() else None
            record = {"event": "heartbeat", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "training_pid": child.pid, "gpu": gpu, "metrics_age_seconds": age,
                      "preprocessing_complete": (args.cache_dir / "train.h5").exists(),
                      "exit_code": child.poll()}
            if age is not None and age > 1800 and record["preprocessing_complete"]:
                record["warning"] = "No metrics update for >30min; inspect, no automatic restart"
            print(json.dumps(record), flush=True)
            with (args.output_dir / "supervision.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if child.poll() is not None:
                success = child.returncode == 0 and (args.output_dir / "test_final.json").exists()
                print(json.dumps({"event": "supervisor_finished", "success": success, "exit_code": child.returncode}), flush=True)
                raise SystemExit(0 if success else 1)
            # Poll at most once a minute without tying up an interactive turn.
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass


if __name__ == "__main__":
    main()
