"""
C3 G2 — Comment 3: how much detection does metadata-only give up? Per RAW CICIDS-2017 label recall of the
deployed model (v2, tau from the policy file) on the v2 test split (primary) and on tune+test (secondary).

  python c3_rawlabel_recall.py --repo <repo> --data <cicids2017 csv dir> --cache <dir with prepared_*.npz> --out <dir>

The raw labels are re-aligned with the committed v2 split exactly as train_pipeline_v2.prepare_data builds it:
load -> map_cicids_to_features / map_cicids_labels -> deduplicate(conflict "drop") -> split indices (committed).
Consistency checks: mapped labels of the re-aligned raw labels == the cached y, and the cached X matches the
re-mapped X on every non-inf cell. Scores come from the deployed .pkl (it reproduces the regression lock exactly, E1b).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    for k in ("repo", "data", "cache", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo)); sys.path.insert(0, str(Path(__file__).resolve().parent))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import joblib

    from c12_common import clopper_pearson
    from edge.inference.decision import get_default_policy
    from edge.training import dataset_loader as dl
    from edge.training.dedup import deduplicate
    from edge.training.feature_mapper import map_cicids_to_features
    from edge.training.label_mapper import CLASS_NAMES, map_cicids_labels

    print("loading CICIDS-2017 CSVs", flush=True)
    df = dl.load_cicids2017(Path(a.data))
    raw = df["Label"].astype(str).str.strip().to_numpy()
    X = map_cicids_to_features(df); y = map_cicids_labels(df["Label"])
    del df
    keep, _ = deduplicate(X, y, "drop")
    X, y, raw = X[keep], y[keep], raw[keep]
    split = np.load(repo / "edge/models_v2/split_indices_v2.npz")
    cached = None
    for c in sorted(Path(a.cache).glob("prepared_*.npz")):
        z = np.load(c)
        if all(np.array_equal(z[f"part_{k}"], split[k]) for k in ("train", "val", "tune", "test")):
            cached = z; break
    if cached is None:
        raise SystemExit("no prepared cache matches the committed split")
    Xc, yc = cached["X"].astype(np.float32), cached["y"]
    checks = {"rows": int(len(y)), "labels_equal_cache": bool(np.array_equal(y, yc))}
    fin = np.isfinite(X) & (X == X)
    diff = np.abs(X.astype(np.float32) - Xc)[fin]
    checks["max_abs_diff_noninf_cells"] = float(diff.max()) if diff.size else 0.0
    print("consistency:", checks, flush=True)
    if not checks["labels_equal_cache"]:
        raise SystemExit("raw-label alignment failed")

    pol = get_default_policy(); tau = pol.alert_threshold
    model = joblib.load(repo / "edge/models_v2/lgbm_classifier.pkl")
    res = {"repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(), "policy": pol.version,
           "tau": tau, "checks": checks, "splits": {}}
    for name, idx in (("test", split["test"]), ("tune_plus_test", np.concatenate([split["tune"], split["test"]]))):
        proba = model.predict_proba(Xc[idx]); cls = list(model.classes_)
        s = 1.0 - proba[:, cls.index(0)]
        att = [i for i, c in enumerate(cls) if c != 0]
        top = np.array(cls)[np.array(att)[np.argmax(proba[:, att], axis=1)]]
        rows = {}
        for lab in sorted(set(raw[idx])):
            m = raw[idx] == lab
            al = s[m] >= tau
            k, n = int(al.sum()), int(m.sum())
            lo, hi = clopper_pearson(k, n)
            sc = int(yc[idx][m][0])
            d = {"saint_class": CLASS_NAMES[sc], "n": n}
            if sc == 0:
                d.update({"alerted": k, "false_alarm_rate": k / n, "ci95": [lo, hi]})
            else:
                named = top[m][al]
                d.update({"detected": k, "recall": k / n, "ci95": [lo, hi],
                          "named_correctly": int((named == sc).sum()),
                          "named_as": {CLASS_NAMES[int(c)]: int(v) for c, v in zip(*np.unique(named, return_counts=True))}})
            rows[lab] = d
        res["splits"][name] = rows
        print(f"== {name}", flush=True)
        for lab, d in rows.items():
            print(f"  {lab[:34]:34s} {d['saint_class']:18s} n={d['n']:>7} "
                  + (f"recall={d['recall']:.4f} ci={[round(x, 4) for x in d['ci95']]} named={d['named_as']}" if 'recall' in d
                     else f"alerted={d['alerted']} ({100 * d['false_alarm_rate']:.3f}%)"), flush=True)
    (out / "c3_rawlabel_recall.json").write_text(json.dumps(res, indent=1, default=str))
    print("written c3_rawlabel_recall.json", flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
