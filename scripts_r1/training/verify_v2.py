"""
Check a finished (or prepared) v2 run before bringing it back.

    python verify_v2.py <repo_dir> [--prepare-only]

Exit code 0 = all checks passed. Standard library only.
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_COMMIT = (HERE / "EXPECTED_COMMIT").read_text().strip()
MANIFEST = HERE / "data" / "dataset_manifest.tsv"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_dataset(data_dir):
    rows = list(csv.DictReader(open(MANIFEST), delimiter="\t"))
    ok = True
    for r in rows:
        p = Path(data_dir) / r["file"]
        good = p.exists() and p.stat().st_size == int(r["size_bytes"]) and sha256(p) == r["sha256"]
        print(f"  [{'OK' if good else 'FAIL'}] {r['file']}")
        ok &= good
    return ok


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--dataset":
        sys.exit(0 if check_dataset(sys.argv[2]) else 1)
    repo = Path(sys.argv[1]).resolve()
    checks = []
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    checks.append(("repo at expected commit " + EXPECTED_COMMIT[:7], head == EXPECTED_COMMIT))
    if "--prepare-only" in sys.argv:
        _report(checks); return
    out = repo / "edge" / "models_v2"
    rep_p = out / "training_report.json"
    if not rep_p.exists():
        checks.append(("training_report.json exists", False)); _report(checks); return
    r = json.loads(rep_p.read_text())
    man = {row["file"]: row["sha256"] for row in csv.DictReader(open(MANIFEST), delimiter="\t")}
    lg = r.get("lightgbm", {})
    checks += [
        ("mode == full", r.get("mode") == "full"),
        ("report git commit == expected", r.get("git", {}).get("commit") == EXPECTED_COMMIT),
        ("no tracked files modified during run", r.get("git", {}).get("tracked_files_modified") is False),
        ("dataset sha256 == manifest", r.get("data", {}).get("dataset_sha256") == man),
        ("leakage checks all 0", all(v == 0 for v in r.get("data", {}).get("leakage_check", {"x": 1}).values())),
        ("50 completed Optuna trials", sum(t.get("value") is not None for t in lg.get("optuna_trials", [])) >= 50),
        ("CPU device (approved protocol)", lg.get("device") == "cpu"),
        ("refit_check ok", lg.get("refit_check", {}).get("ok") is True),
        ("control run present", "control_v1_params" in r),
        ("ONNX export gates passed", bool(r.get("onnx_export_gates")) and all(r["onnx_export_gates"].values())),
        ("ONNX FP32 argmax agreement >= 0.999",
         r.get("onnx_fidelity_real_test", {}).get("lgbm_classifier.onnx", {}).get("argmax_agreement", 0) >= 0.999),
    ]
    for f in ["lgbm_classifier.pkl", "isolation_forest.pkl", "lgbm_classifier.onnx", "lgbm_classifier_int8.onnx",
              "lgbm_background_100.npy", "if_background_100.npy", "split_indices_v2.npz", "training_curves_v2.json"]:
        checks.append((f"artifact {f}", (out / f).exists()))
    _report(checks)
    t = r.get("evaluation", {}).get("test", {})
    print(f"\n  test F1w={t.get('f1_weighted')}  FPR={t.get('false_positive_rate')}  "
          f"elapsed={r.get('elapsed_s')} s  (informational, not a pass/fail gate)")


def _report(checks):
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok in checks if not ok]
    print(f"\n{'ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
