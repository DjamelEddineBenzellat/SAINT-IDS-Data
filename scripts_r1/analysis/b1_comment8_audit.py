"""
B1 — Reviewer 1 Comment 8: data-integrity and leakage audit.

Reproduces the V1 preprocessing + split by IMPORTING the original functions
(no source edits; only the dataset directory is redirected at runtime), then
measures duplicates and cross-partition overlap. Uses the saved V1 LightGBM
model only for evaluation (no training). Writes outputs only under
revision-audit/.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/path/to/SAINT/realtime-ids-healthcareBeta")
AUD = Path("/path/to/SAINT/revision-audit")
DATA = ROOT / "data" / "cicids2017"
OUT = AUD / "reports" / "comment8"
WORK = AUD / "work"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import edge.training.dataset_loader as dl  # noqa: E402
from edge.training.feature_mapper import map_cicids_to_features  # noqa: E402
from edge.training.label_mapper import CLASS_NAMES, NUM_CLASSES, map_cicids_labels  # noqa: E402
from edge.training.train_pipeline import balance_classes  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

T0 = time.time()
LOG = []


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


def row_hash(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


summary: dict = {"script": __file__, "dataset_dir": str(DATA)}

# ── 1. Load with the ORIGINAL loader (directory redirected at runtime) ───────
dl.CICIDS2017_DIR = DATA
df = dl.load_cicids2017()
log(f"loaded raw: {df.shape}")
manifest = pd.read_csv(AUD / "reports" / "dataset_manifest.tsv", sep="\t").set_index("file")
file_rows = [int(manifest.loc[f, "rows_excl_header"]) for f in dl.CICIDS2017_FILES]
assert sum(file_rows) == len(df), "row count mismatch with dataset manifest"
file_id = np.repeat(np.arange(len(dl.CICIDS2017_FILES)), file_rows)
summary["files_in_load_order"] = dl.CICIDS2017_FILES
summary["rows_per_file"] = dict(zip(dl.CICIDS2017_FILES, file_rows))
raw_label = df["Label"].astype(str).str.strip().to_numpy()
dst_port_raw = df["Destination Port"].to_numpy()
summary["raw_columns"] = int(df.shape[1])
summary["columns_absent"] = [c for c in ["Timestamp", "Protocol", "Source IP", "Destination IP",
                                         "Flow ID", "Source Port"] if c not in df.columns]

# ── 2. Raw-level duplicates (all 79 columns, and 78 features without Label) ──
feat_cols = [c for c in df.columns if c != "Label"]
num = df[feat_cols].apply(pd.to_numeric, errors="coerce")
summary["raw_nonnumeric_cells_coerced"] = int(num.isna().sum().sum() - df[feat_cols].isna().sum().sum())
inf_counts = {c: int(np.isinf(num[c].to_numpy(dtype=float)).sum()) for c in feat_cols}
summary["raw_inf_cells_by_column"] = {k: v for k, v in inf_counts.items() if v}
summary["raw_nan_cells_by_column"] = {c: int(v) for c, v in df[feat_cols].isna().sum().items() if v}
h_raw_feat = row_hash(df[feat_cols])
h_raw_full = row_hash(pd.DataFrame({"f": h_raw_feat, "l": raw_label}))
del num


def dup_stats(h: np.ndarray, labels: np.ndarray) -> dict:
    s = pd.Series(h)
    dup_extra = int(s.duplicated(keep="first").sum())
    n_unique = int(s.nunique())
    g = pd.DataFrame({"h": h, "y": labels}).groupby("h")["y"].nunique()
    conflict_hashes = set(g[g > 1].index)
    rows_in_conflict = int(np.isin(h, list(conflict_hashes)).sum()) if conflict_hashes else 0
    return {"rows": int(len(h)), "unique": n_unique, "redundant_copies": dup_extra,
            "redundant_pct": round(100 * dup_extra / len(h), 3),
            "rows_with_any_duplicate": int(s.duplicated(keep=False).sum()),
            "label_conflict_groups": len(conflict_hashes), "rows_in_label_conflict_groups": rows_in_conflict}


summary["raw_dup_all79_cols"] = dup_stats(h_raw_full, raw_label)
summary["raw_dup_78_features"] = dup_stats(h_raw_feat, raw_label)
log(f"raw dups: {summary['raw_dup_all79_cols']['redundant_copies']} (79 cols), "
    f"{summary['raw_dup_78_features']['redundant_copies']} (78 feat)")

# Inf handling leak check (feature_mapper._clean_infinites uses FULL-data column max)
raw_inf_info = {}
for c in ["Flow Bytes/s", "Flow Packets/s"]:
    col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    raw_inf_info[c] = {"inf_cells": int(np.isinf(col).sum()), "nan_cells": int(np.isnan(col).sum())}
summary["mapped_source_inf_nan"] = raw_inf_info

# ── 3. ORIGINAL mapping, labels, balancing ─────────────────────────────────
X = map_cicids_to_features(df)
y = map_cicids_labels(df["Label"])
del df
log(f"mapped X {X.shape} {X.dtype}")
constant_cols = [int(i) for i in range(X.shape[1]) if np.all(X[:, i] == X[0, i])]
summary["mapped_constant_feature_indices"] = constant_cols
Xb, yb = balance_classes(X, y)
summary["balance_classes_is_identity"] = bool(Xb.shape == X.shape and np.array_equal(yb, y)
                                               and np.array_equal(Xb, X))
log(f"balance_classes identity: {summary['balance_classes_is_identity']}")

# ── 4. ORIGINAL split (indices carried alongside; permutation depends only on
#        n_samples, stratify, random_state) ─────────────────────────────────
idx = np.arange(len(yb))
i_train, i_temp, y_train, y_temp = train_test_split(idx, yb, test_size=0.30, stratify=yb, random_state=42)
i_val, i_test, y_val, y_test = train_test_split(i_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
i_trval = np.concatenate([i_train, i_val])
summary["split_sizes"] = {"train": len(i_train), "val": len(i_val), "test": len(i_test)}
log(f"split sizes {summary['split_sizes']}")

# ── 5. Mapped-space (33-feature model input) duplicates & overlap ───────────
hX = row_hash(pd.DataFrame(X))
hXy = row_hash(pd.DataFrame({"h": hX, "y": y}))
summary["mapped_dup_33_features"] = dup_stats(hX, y)


def overlap(h_src: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Boolean mask over dst_idx: row's hash appears somewhere in src_idx."""
    return np.isin(h_src[dst_idx], np.unique(h_src[src_idx]))


ov = {}
for space, h in [("raw78", h_raw_feat), ("raw79_with_label", h_raw_full),
                 ("mapped33", hX), ("mapped33_with_label", hXy)]:
    m_test_tr = overlap(h, i_train, i_test)
    m_test_trval = overlap(h, i_trval, i_test)
    m_val_tr = overlap(h, i_train, i_val)
    ov[space] = {"test_rows_seen_in_train": int(m_test_tr.sum()),
                 "test_rows_seen_in_train_or_val": int(m_test_trval.sum()),
                 "test_pct_seen_in_train_or_val": round(100 * m_test_trval.mean(), 3),
                 "val_rows_seen_in_train": int(m_val_tr.sum()),
                 "val_pct_seen_in_train": round(100 * m_val_tr.mean(), 3)}
summary["cross_partition_overlap"] = ov
# same 33-vector across partitions but with DIFFERENT label (vector seen, label pair never seen)
seen_vec = overlap(hX, i_trval, i_test)
seen_pair = overlap(hXy, i_trval, i_test)
summary["test_rows_vector_seen_label_differs_only"] = int((seen_vec & ~seen_pair).sum())
log(f"overlap mapped33 test∩trval = {ov['mapped33']['test_pct_seen_in_train_or_val']}%")

# per-class overlap (mapped33 with label, vs train∪val = final model's training data)
per_class = []
for k in range(NUM_CLASSES):
    m = y_test == k
    n = int(m.sum())
    per_class.append({"class": CLASS_NAMES[k], "test_rows": n,
                      "seen_vec_in_trval": int(seen_vec[m].sum()),
                      "seen_vec_pct": round(100 * seen_vec[m].mean(), 2) if n else None})
summary["test_overlap_per_class_mapped33"] = per_class

# ── 6. Inf-replacement statistic: full-data max vs train-only max ───────────
# feature_mapper replaces ±inf by the column max of finite values over ALL rows.
# Check whether that max is attained inside the training partition.
inf_leak = {}
for name, fidx in [("packets_per_second(idx3)", 3), ("connections_per_src_ip_per_minute(idx23)", 23)]:
    col = X[:, fidx]
    full_max = float(col.max())
    tr_max = float(col[i_train].max())
    inf_leak[name] = {"full_data_max_used": full_max, "train_only_max": tr_max,
                      "identical": bool(full_max == tr_max)}
summary["inf_replacement_statistic_check"] = inf_leak

# ── 7. Re-evaluate the SAVED V1 LightGBM on the reproduced test set ─────────
import joblib  # noqa: E402

lgbm = joblib.load(ROOT / "edge" / "models" / "lgbm_classifier.pkl")
Xte = X[i_test]
t1 = time.time()
proba = lgbm.predict_proba(Xte)
pred = lgbm.classes_[np.argmax(proba, axis=1)]
log(f"predict_proba on {len(Xte)} rows in {time.time() - t1:.1f}s")
labels6 = list(range(NUM_CLASSES))
cm = confusion_matrix(y_test, pred, labels=labels6)
report = json.loads((ROOT / "edge" / "models" / "training_report.json").read_text())
cm_v1 = np.array(report["lgbm_metrics"]["confusion_matrix"])
summary["reproduced_confusion_matrix"] = cm.tolist()
summary["confusion_matrix_equals_training_report"] = bool(np.array_equal(cm, cm_v1))
summary["test_distribution"] = {CLASS_NAMES[k]: int((y_test == k).sum()) for k in labels6}


def metrics(yt, yp):
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels6, zero_division=0.0)
    bt, bp = yt != 0, yp != 0
    fp = int(np.sum(bp & ~bt))
    tn = int(np.sum(~bp & ~bt))
    return {"n": int(len(yt)), "f1_weighted": round(float(f1_score(yt, yp, average="weighted", labels=labels6, zero_division=0.0)), 6),
            "fpr": round(fp / (fp + tn), 6) if fp + tn else None,
            "per_class": {CLASS_NAMES[k]: {"support": int(s[k]), "precision": round(float(p[k]), 4),
                                           "recall": round(float(r[k]), 4), "f1": round(float(f[k]), 4)}
                          for k in labels6 if s[k] > 0}}


unseen = ~seen_vec
# unseen AND keep one copy per distinct (vector,label) inside the test set
hXy_te = hXy[i_test]
first_occ = ~pd.Series(hXy_te).duplicated(keep="first").to_numpy()
summary["eval_full_test"] = metrics(y_test, pred)
summary["eval_test_unseen_vectors"] = metrics(y_test[unseen], pred[unseen])
summary["eval_test_unseen_and_dedup"] = metrics(y_test[unseen & first_occ], pred[unseen & first_occ])
summary["eval_test_dedup_only"] = metrics(y_test[first_occ], pred[first_occ])
log(f"F1 full={summary['eval_full_test']['f1_weighted']} unseen={summary['eval_test_unseen_vectors']['f1_weighted']} "
    f"unseen+dedup={summary['eval_test_unseen_and_dedup']['f1_weighted']}")

# ── 8. Grouping feasibility: class × capture file ──────────────────────────
cxf = pd.crosstab(pd.Series(y, name="class").map(dict(enumerate(CLASS_NAMES))),
                  pd.Series(file_id, name="file").map(dict(enumerate(dl.CICIDS2017_FILES))))
cxf.to_csv(OUT / "comment8_class_by_file.csv")
raw_lab_file = pd.crosstab(pd.Series(raw_label, name="raw_label"),
                           pd.Series(file_id, name="file").map(dict(enumerate(dl.CICIDS2017_FILES))))
raw_lab_file.to_csv(OUT / "comment8_rawlabel_by_file.csv")
# duplicates spanning different capture files (mapped space)
g = pd.DataFrame({"h": hX, "f": file_id}).groupby("h")["f"].nunique()
summary["mapped33_vectors_spanning_multiple_files"] = int((g > 1).sum())

# ── 9. Class counts at each stage ───────────────────────────────────────────
rows = []
def add(stage, labels_arr):
    c = Counter(labels_arr.tolist())
    rows.append([stage] + [int(c.get(k, 0)) for k in labels6] + [int(len(labels_arr))])
add("raw_mapped_labels", y)
add("after_balance_classes", yb)
add("train", y_train); add("val", y_val); add("test", y_test)
add("unique_mapped33_vectors_first_occurrence", y[~pd.Series(hXy).duplicated(keep="first").to_numpy()])
add("test_unseen_vectors", y_test[unseen])
add("test_unseen_and_dedup", y_test[unseen & first_occ])
with open(OUT / "comment8_class_counts.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["stage"] + CLASS_NAMES + ["total"])
    w.writerows(rows)
summary["raw_label_counts"] = {k: int(v) for k, v in Counter(raw_label.tolist()).most_common()}

# ── 10. Save reusable split artefacts for B2 (audit workspace only) ─────────
np.savez_compressed(WORK / "v1_split_reproduction.npz", i_train=i_train, i_val=i_val, i_test=i_test,
                    y=y, file_id=file_id, X_test=Xte, dst_port_test=dst_port_raw[i_test].astype(np.int64),
                    seen_vec_test=seen_vec)
summary["elapsed_s"] = round(time.time() - T0, 1)
summary["log"] = LOG
(OUT / "comment8_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
log("done")
