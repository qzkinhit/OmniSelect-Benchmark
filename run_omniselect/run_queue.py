"""Run the lines of queue files on this machine, one after another or in parallel on several GPUs.

A queue line is one shell command, optionally followed by ``# cell=<run record dir> name=<job>``.
Blank lines and lines that start with ``#`` are ignored. A line whose cell already holds
decision.json is reported as complete and not started, so an interrupted queue can be restarted.
Every job runs from the repository root and writes its output to ``<log-dir>/<name>.log``.

With ``--gpus 0,1`` the jobs are spread over the listed devices through CUDA_VISIBLE_DEVICES, at
most ``--jobs-per-gpu`` at a time on each device. Without ``--gpus``, ``--jobs`` jobs run at a time
and see the environment unchanged. ``--threads N`` sets OMP_NUM_THREADS, MKL_NUM_THREADS and
OPENBLAS_NUM_THREADS of every job. Exit code 3 of the driver (model or dataset unavailable) is
reported as skipped. The runner exits with 1 when any job failed.

    python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt
    python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt --gpus 0,1 --jobs-per-gpu 2 --threads 4
    python -m run_omniselect.run_queue run_omniselect/queues/ablation_v2_3.txt --dry-run
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
META = re.compile(r"\s#\s*(.*)$")
SKIP_CODE = 3


@dataclass
class Job:
    """One queue line: the command, the job name and the run-record directory it writes."""

    command: str
    name: str
    cell: Optional[str]
    source: str


def parse_queue(path: Path) -> list[Job]:
    """Jobs of a queue file in file order. Names default to <file stem>_<line number>."""
    jobs = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        meta: dict[str, str] = {}
        match = META.search(line)
        if match:
            meta = dict(item.split("=", 1) for item in match.group(1).split() if "=" in item)
        name = meta.get("name") or f"{path.stem}_{number}"
        jobs.append(Job(command=line, name=name, cell=meta.get("cell"), source=f"{path}:{number}"))
    return jobs


def is_complete(job: Job, root: Path) -> bool:
    """True when the job's cell already holds decision.json."""
    if not job.cell:
        return False
    cell = Path(job.cell)
    return (cell if cell.is_absolute() else root / cell).joinpath("decision.json").is_file()


def job_env(device: Optional[str], threads: Optional[int]) -> dict[str, str]:
    """Environment of one job: the device through CUDA_VISIBLE_DEVICES and the thread counts."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[var] = str(threads)
    return env


def run_job(job: Job, device: Optional[str], threads: Optional[int], log_dir: Path) -> int:
    """Run one job with bash from the repository root and return its exit code."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{job.name}.log", "a") as log:
        log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {job.source} device={device}\n# {job.command}\n")
        log.flush()
        proc = subprocess.run(["bash", "-c", job.command], cwd=REPO_ROOT, env=job_env(device, threads),
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    """Run every pending job of the given queue files and print one status line per job."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queues", nargs="+", type=Path)
    parser.add_argument("--gpus", default="", help="comma-separated device ids, for example 0,1")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1, help="parallel jobs without --gpus")
    parser.add_argument("--threads", type=int, default=None, help="BLAS and OpenMP threads per job")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results_and_logs" / "logs")
    parser.add_argument("--dry-run", action="store_true", help="list the pending jobs without running them")
    args = parser.parse_args(argv)

    jobs = [job for path in args.queues for job in parse_queue(path)]
    pending = [job for job in jobs if not is_complete(job, REPO_ROOT)]
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} complete, {len(pending)} pending")
    if args.dry_run:
        for job in pending:
            print(f"{job.name}  {job.command}")
        return 0

    devices = [d.strip() for d in args.gpus.split(",") if d.strip()]
    slots: "queue.Queue[Optional[str]]" = queue.Queue()
    for device in devices:
        for _ in range(max(1, args.jobs_per_gpu)):
            slots.put(device)
    if not devices:
        for _ in range(max(1, args.jobs)):
            slots.put(None)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    def work(job: Job) -> None:
        device = slots.get()
        try:
            code = run_job(job, device, args.threads, args.log_dir)
        finally:
            slots.put(device)
        status = "done" if code == 0 else "skipped" if code == SKIP_CODE else "failed"
        with lock:
            counts[status] += 1
            print(f"[{status}] {job.name} (exit {code}, device {device}, log {args.log_dir / (job.name + '.log')})",
                  flush=True)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
        list(pool.map(work, pending))
    print(f"finished: {counts['done']} done, {counts['skipped']} skipped, {counts['failed']} failed")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
