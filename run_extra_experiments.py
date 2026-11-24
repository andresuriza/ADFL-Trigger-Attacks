#!/usr/bin/env python3
"""Parallel launcher for the three extra ADFL experiments.

Experiments (all executed from ``adfl-clean.ipynb`` via env vars only):

* exp1_partition_var          ADFL_RUN=5  Fashion-MNIST Non-IID partition/teacher variance
* exp2_teacher_gan_ablation   ADFL_RUN=6  Fashion-MNIST Non-IID teacher/GAN 2x2 ablation
* exp3_cifar_multiseed        ADFL_RUN=7  CIFAR-10 IID headline, seeds 42,43,44

Each run gets an isolated work dir ``runs/<label>/work`` into which the base
checkpoints it needs are staged from ``--stage-from`` (default ``box-insurance``)
so the notebook's ``load_or_train`` reuses them. Runs execute concurrently on a
single GPU, each pinned to a disjoint CPU core range (Linux only) with
OMP/MKL/OPENBLAS threads = 1, because FL training here is single-thread
launch-bound. Result JSONs are collected into ``results_extra/``.

Usage:
    python run_extra_experiments.py --dry-run
    python run_extra_experiments.py --smoke
    python run_extra_experiments.py --jobs 3 --cores-per-run 8
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = "adfl-clean.ipynb"

FASHION_NEEDS = [
    "backdoored_fashion_mnist_dirichlet0.5_*_paper_dirichlet*.pt",
    "purified_fashion_mnist_dirichlet0.5_*_paper_dirichlet*.pt",  # lets seed_rows()/full-FL confirm resume from cache
    "clean_fashion_mnist_dirichlet0.5_paper_dirichlet.pt",
    "clean_fashion_mnist_paper_iid.pt",
    "generator_fashion_mnist*_paper*.pt",
    "distill_fashion_mnist*_paper*.pt",
    "results_fashion_mnist_dirichlet0.5.json",
    "multiseed_fashion_mnist_dirichlet0.5.json",
    "clean_sweep_fashion_mnist_dirichlet0.5.json",               # present -> the clean-budget sweep cell self-skips
]
CIFAR_NEEDS = [
    "backdoored_cifar10_resnet18_pixel_paper_iid.pt",
    "backdoored_cifar10_resnet18_semantic_paper_iid.pt",
    "purified_cifar10_resnet18_*_paper_iid.pt",                  # seed-42 purified -> seed_rows(42) returns cached
    "clean_cifar10_resnet18_paper_iid.pt",
    "generator_cifar10_resnet18_paper_iid.pt",
    "distill_cifar10_resnet18_paper_iid.pt",
    "results_cifar10_resnet18.json",
    "kd_lr_sweep_cifar10_resnet18.json",
    "clean_sweep_cifar10_resnet18.json",
]


def _verify_exp3(work: Path) -> tuple[bool, str]:
    """exp3 stages results_cifar10_resnet18.json IN and writes it OUT, so file-presence proves
    nothing. Confirm the multi-seed merge actually ran: the pixel/semantic rows must carry a
    3-seed aggregate (n_seeds>=3 with an asr_post_std field)."""
    import json
    f = work / "results_cifar10_resnet18.json"
    if not f.exists():
        return False, "results_cifar10_resnet18.json missing"
    try:
        data = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable results json: {e}"
    rows = {r.get("attack"): r for r in data.get("attacks", [])}
    bad = [a for a in ("pixel", "semantic")
           if not (rows.get(a, {}).get("n_seeds", 0) >= 3 and "asr_post_std" in rows.get(a, {}))]
    return (not bad), ("3-seed aggregate present for pixel+semantic" if not bad
                       else f"no 3-seed aggregate for: {bad}")

def _verify_mnist_noniid(work: Path) -> tuple[bool, str]:
    """MNIST Non-IID negative control: the results file must carry a 3-seed pixel aggregate."""
    import json
    f = work / "results_mnist_dirichlet0.5.json"
    if not f.exists():
        return False, "results_mnist_dirichlet0.5.json missing"
    try:
        data = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable results json: {e}"
    rows = {r.get("attack"): r for r in data.get("attacks", [])}
    ok = rows.get("pixel", {}).get("n_seeds", 0) >= 3 and "asr_post_std" in rows.get("pixel", {})
    return ok, ("3-seed aggregate present for pixel" if ok else "no 3-seed pixel aggregate")


def _verify_adaptive(work: Path) -> tuple[bool, str]:
    """Adaptive-trigger attacker: the output is written fresh (not staged), so confirm the search +
    full-FL confirm actually produced at least one mode with a full-FL ASR and an off-manifold
    coverage number."""
    import json
    f = work / "adaptive_trigger_fashion_mnist_dirichlet0.5.json"
    if not f.exists():
        return False, "adaptive_trigger_fashion_mnist_dirichlet0.5.json missing"
    try:
        data = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable results json: {e}"
    res = data.get("results", [])
    good = [r for r in res if "asr_post" in r and r.get("coverage_mean") is not None]
    if not good:
        return False, "no mode produced a full-FL asr_post + coverage"
    best = max(good, key=lambda r: r.get("asr_post", 0.0))
    return True, (f"{len(good)} mode(s); best={best.get('mode')} asr_post={best.get('asr_post'):.3f} "
                  f"(static baseline survivor ~0.43)")


# Order matters: the CIFAR run is the wall-clock long pole, so it goes first.
RUN_MATRIX = [
    dict(label="exp3_cifar_multiseed", run=7, experiment="",
         work_dir="runs/exp3_cifar_multiseed/work", needs=CIFAR_NEEDS,
         outputs=["results_cifar10_resnet18.json",
                  "backdoored_cifar10_resnet18_*_paper_iid_s*.pt"],
         verify=_verify_exp3),
    dict(label="exp1_partition_var", run=5, experiment="partition_var",
         work_dir="runs/exp1_partition_var/work", needs=FASHION_NEEDS,
         outputs=["partition_var_fashion_mnist_dirichlet0.5.json"]),
    dict(label="exp2_teacher_gan_ablation", run=6, experiment="teacher_gan_ablation",
         work_dir="runs/exp2_teacher_gan_ablation/work", needs=FASHION_NEEDS,
         outputs=["teacher_gan_ablation_fashion_mnist_dirichlet0.5.json"]),
    # MNIST Non-IID negative control. Nothing to stage (box-insurance has only MNIST IID); trains
    # fresh. experiment="mnist_noniid_control" is a non-matching sentinel: it trips the evasion-screen
    # skip-guards (if not EXPERIMENT) without triggering any gated experiment cell, keeping this a
    # bounded multi-seed backdoor-removal control (pixel is MNIST's only registered attack).
    dict(label="exp4_mnist_noniid", run=8, experiment="mnist_noniid_control",
         work_dir="runs/exp4_mnist_noniid/work", needs=[],
         outputs=["results_mnist_dirichlet0.5.json", "multiseed_mnist_dirichlet0.5.json"],
         verify=_verify_mnist_noniid),
    # Adaptive-trigger attacker vs ADFL (Exp 5). Stages the Fashion Non-IID set so the primary
    # backdoored model + GAN + teacher + distillation set are reused (the search runs on top of
    # them); the per-mode full-FL confirm still trains a fresh backdoored model for the found
    # trigger. experiment="adaptive_trigger" runs the gated ES cell (RUN=9) and skips the standard
    # evasion screen. ADV_* env knobs (gens/lambda/modes) can tune the search budget.
    dict(label="exp5_adaptive", run=9, experiment="adaptive_trigger",
         work_dir="runs/exp5_adaptive/work", needs=FASHION_NEEDS,
         outputs=["adaptive_trigger_fashion_mnist_dirichlet0.5.json"],
         verify=_verify_adaptive),
]


def run_dir(spec: dict) -> Path:
    return ROOT / "runs" / spec["label"]


def available_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def assign_cores(index: int, cores_per_run: int, total: int) -> list[int] | None:
    """Disjoint core range for the index-th run (None if pinning unsupported)."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    pool = sorted(os.sched_getaffinity(0))
    start = (index * cores_per_run) % max(len(pool), 1)
    chosen = pool[start:start + cores_per_run]
    return chosen or pool


def build_env(spec: dict, data_dir: Path, smoke: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "ADFL_RUN": str(spec["run"]),
        "ADFL_EXPERIMENT": spec["experiment"],
        "ADFL_WORK_DIR": str((ROOT / spec["work_dir"]).resolve()),
        "ADFL_DATA_DIR": str(data_dir),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    if smoke:
        env["ADFL_FIDELITY"] = "fast"
    else:
        env.pop("ADFL_FIDELITY", None)  # unset = paper tier
    return env


def warm_datasets(data_dir: Path) -> None:
    """Download all datasets once, sequentially, into the shared cache."""
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torchvision.datasets as tvd
    except ImportError:
        print("[warn] torchvision not importable; skipping dataset pre-download "
              "(notebooks will download on demand and may race).")
        return
    for name in ("MNIST", "FashionMNIST", "CIFAR10"):
        print(f"[data] fetching {name} -> {data_dir}")
        for train in (True, False):
            getattr(tvd, name)(root=str(data_dir), train=train, download=True)


def stage(spec: dict, stage_from: Path) -> None:
    work = ROOT / spec["work_dir"]
    work.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in spec["needs"]:
        matches = sorted(glob.glob(str(stage_from / pattern)))
        if not matches:
            print(f"[warn] {spec['label']}: no match for '{pattern}' in {stage_from}")
            continue
        for m in matches:
            shutil.copy2(m, work / Path(m).name)
            copied += 1
    print(f"[stage] {spec['label']}: copied {copied} file(s) into {work}")


def build_command(out_nb: Path) -> list[str]:
    if importlib.util.find_spec("papermill") is not None:
        return [sys.executable, "-m", "papermill", NOTEBOOK, str(out_nb)]
    return [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--ExecutePreprocessor.timeout=-1",
            "--output", out_nb.name, "--output-dir", str(out_nb.parent), NOTEBOOK]


def launch(spec: dict, env: dict, cores: list[int] | None):
    rdir = run_dir(spec)
    rdir.mkdir(parents=True, exist_ok=True)
    out_nb = rdir / f"executed_{spec['label']}.ipynb"
    log = open(rdir / "log.txt", "w", encoding="utf-8")
    kwargs = {}
    if cores and hasattr(os, "sched_setaffinity"):
        kwargs["preexec_fn"] = lambda c=set(cores): os.sched_setaffinity(0, c)
    proc = subprocess.Popen(build_command(out_nb), cwd=str(ROOT), env=env,
                            stdout=log, stderr=subprocess.STDOUT, **kwargs)
    return proc, log


def collect(spec: dict, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    work = ROOT / spec["work_dir"]
    found = []
    for pattern in spec["outputs"]:
        for m in sorted(glob.glob(str(work / pattern))):
            shutil.copy2(m, dest / Path(m).name)
            found.append(Path(m).name)
    return found


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="set ADFL_FIDELITY=fast for every run (quick end-to-end check)")
    ap.add_argument("--only", action="append", default=[], metavar="LABEL",
                    help="run only this label (repeatable)")
    ap.add_argument("--stage-from", default="box-insurance",
                    help="dir holding base checkpoints (default: box-insurance)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="max concurrent runs (default: min(#runs, cores // cores-per-run))")
    ap.add_argument("--cores-per-run", type=int, default=8,
                    help="CPU cores reserved per run (Linux pinning; default 8)")
    ap.add_argument("--data-dir", default="data",
                    help="shared torchvision dataset cache (default: ./data)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and env per run; stage and launch nothing")
    return ap.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def main() -> int:
    args = parse_args()
    labels = {s["label"] for s in RUN_MATRIX}
    unknown = set(args.only) - labels
    if unknown:
        print(f"Unknown label(s): {sorted(unknown)}; valid: {sorted(labels)}")
        return 2
    matrix = [s for s in RUN_MATRIX if not args.only or s["label"] in args.only]

    stage_from = resolve(args.stage_from)
    data_dir = resolve(args.data_dir).resolve()
    results_dir = ROOT / "results_extra"
    jobs = args.jobs or min(len(matrix), max(1, available_cores() // args.cores_per_run))
    jobs = max(1, jobs)
    pinning = hasattr(os, "sched_setaffinity")

    print(f"Plan: {len(matrix)} run(s), jobs={jobs}, cores/run={args.cores_per_run}, "
          f"smoke={args.smoke}, pinning={'on' if pinning else 'off (not Linux)'}")
    print(f"stage-from={stage_from}  data-dir={data_dir}  results={results_dir}")
    print(f"command: {' '.join(build_command(Path('<out>.ipynb')))}")

    cores_for = {s["label"]: assign_cores(i, args.cores_per_run, len(matrix))
                 for i, s in enumerate(matrix)}
    for s in matrix:
        env = build_env(s, data_dir, args.smoke)
        print(f"\n[{s['label']}] run={s['run']} experiment={s['experiment']!r}")
        print(f"  work_dir : {ROOT / s['work_dir']}")
        print(f"  stage    : {s['needs']}")
        print(f"  expects  : {s['outputs']}")
        print(f"  cores    : {cores_for[s['label']]}")
        for k in ("ADFL_RUN", "ADFL_EXPERIMENT", "ADFL_WORK_DIR", "ADFL_DATA_DIR",
                  "ADFL_FIDELITY", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES"):
            print(f"  env {k}={env.get(k, '<unset>')}")
    if args.dry_run:
        print("\n[dry-run] nothing staged or launched.")
        return 0

    if not stage_from.is_dir():
        print(f"[error] stage-from dir not found: {stage_from}")
        return 2
    warm_datasets(data_dir)
    for s in matrix:
        stage(s, stage_from)

    pending = list(matrix)
    running: dict[str, tuple] = {}
    summary: dict[str, dict] = {}
    t0 = time.time()
    try:
        while pending or running:
            while pending and len(running) < jobs:
                s = pending.pop(0)
                proc, log = launch(s, build_env(s, data_dir, args.smoke),
                                   cores_for[s["label"]])
                running[s["label"]] = (s, proc, log, time.time())
                print(f"[launch] {s['label']} pid={proc.pid} log={run_dir(s) / 'log.txt'}")
            for label, (s, proc, log, start) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                del running[label]
                found = collect(s, results_dir)
                ok, note = s.get("verify", lambda w: (True, ""))(ROOT / s["work_dir"])
                summary[label] = dict(rc=rc, elapsed=time.time() - start, found=found,
                                      verify_ok=ok, verify_note=note)
                print(f"[done] {label} rc={rc} in {summary[label]['elapsed']:.0f}s"
                      + (f"  [verify] {note}" if note else ""))
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[interrupt] terminating children")
        for s, proc, log, _ in running.values():
            proc.terminate()
        return 130

    print(f"\n=== Summary (total {time.time() - t0:.0f}s) ===")
    failed = False
    for label, r in summary.items():
        failed |= r["rc"] != 0
        vnote = "" if r.get("verify_ok", True) else f"  VERIFY-FAILED: {r.get('verify_note')}"
        print(f"{label}: rc={r['rc']} elapsed={r['elapsed']:.0f}s "
              f"outputs={r['found'] or 'NONE FOUND'}{vnote}")
        if not r["found"] or not r.get("verify_ok", True):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
