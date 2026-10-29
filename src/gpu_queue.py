"""Minimal GPU job queue for a shared host.

usage: python src/gpu_queue.py JOBFILE [MAX_RUNNING]
JOBFILE: one job per line: "<need_gb> <done_marker_path> <command...>". Jobs whose done marker exists are skipped.
A command may start with "REQUIRES=<path>" - the job is held until that file exists.
Launches the next pending job on the GPU with the most free memory if it has >= need_gb + 1 GB free, at most one
launch per LAUNCH_GAP seconds; a job that exits without its done marker is retried (up to 3 times).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_GAP = 25


def free_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) / 1024 for a, b in (l.split(",") for l in out.strip().splitlines())}


def main(jobfile, max_running=8):
    jobs = []
    for line in open(jobfile):
        if line.strip() and not line.startswith("#"):
            need, marker, cmd = line.strip().split(" ", 2)
            req = None
            if cmd.startswith("REQUIRES="):
                req, cmd = cmd.split(" ", 1)
                req = ROOT / req.split("=", 1)[1]
            jobs.append({"need": float(need), "marker": ROOT / marker, "cmd": cmd, "tries": 0, "req": req})
    running = {}
    env = dict(os.environ, NUMPY_MADVISE_HUGEPAGE="0", OMP_NUM_THREADS="2")
    last = 0
    while True:
        for j, p in list(running.items()):
            if p.poll() is not None:
                del running[j]
                ok = jobs[j]["marker"].exists()
                print(time.strftime("%H:%M:%S"), "finished" if ok else f"FAILED rc={p.returncode}", jobs[j]["cmd"],
                      flush=True)
        todo = [i for i, j in enumerate(jobs) if not j["marker"].exists() and i not in running and j["tries"] < 3]
        pending = [i for i in todo if jobs[i]["req"] is None or jobs[i]["req"].exists()]
        if not todo and not running:
            print("all done", flush=True)
            return
        if pending and len(running) < max_running and time.time() - last > LAUNCH_GAP:
            fm = free_mem()
            g = max(fm, key=fm.get)
            i = pending[0]
            if fm[g] >= jobs[i]["need"] + 1.0:
                jobs[i]["tries"] += 1
                log = ROOT / "logs" / "queue" / (Path(jobs[i]["marker"]).parent.parent.name + "_" + Path(jobs[i]["marker"]).parent.name + f".try{jobs[i]['tries']}.log")
                log.parent.mkdir(parents=True, exist_ok=True)
                running[i] = subprocess.Popen(jobs[i]["cmd"], shell=True, cwd=ROOT, stdout=open(log, "w"),
                                              stderr=subprocess.STDOUT, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)))
                print(time.strftime("%H:%M:%S"), f"launched on GPU{g} ({fm[g]:.1f} GB free):", jobs[i]["cmd"],
                      flush=True)
                last = time.time()
        time.sleep(5)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
