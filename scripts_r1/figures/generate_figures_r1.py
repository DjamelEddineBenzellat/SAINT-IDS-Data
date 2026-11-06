"""
Figures 3-5 and Table 4 for manuscript R1, from the v2 (deduplicated) model.

Reads only (nothing is retrained):
  edge/models_v2/training_report.json   -> Fig. 3 (confusion matrix), Fig. 4 (per-class metrics)
  edge/models_v2/training_curves_v2.json -> Fig. 5 (training curves, same run and split)
  edge/models_v2/lgbm_classifier.pkl     -> Table 4 (split and gain importance); its SHA-256 must
                                            match edge/inference/alert_policy.json (the served model)

Writes to CCPE_revision1/figures/:
  confusion_matrix_v2.{pdf,png}, per_class_metrics_v2.{pdf,png}, training_curves_v2.{pdf,png},
  table4_feature_importance_v2.json, table4_feature_importance_v2.tex

Usage (from the repository root, any Python with matplotlib, numpy, lightgbm, joblib):
  python realtime-ids-healthcare-paper/CCPE_revision1/generate_figures_r1.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_MODELS = _ROOT / "edge" / "models_v2"
_POLICY = _ROOT / "edge" / "inference" / "alert_policy.json"
_OUT = _HERE / "figures"

sys.path.insert(0, str(_ROOT))
from contracts.flow_feature_vector import FEATURE_NAMES  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
})

# No creation dates or software tags, so re-running gives byte-identical files.
_META = {"pdf": {"CreationDate": None, "Producer": None, "Creator": None}, "png": {"Software": None}}

CLASSES = ["BENIGN", "PORT_SCAN", "BRUTE_FORCE", "MITM", "DOS", "DATA_EXFILTRATION"]
SHORT = ["BEN", "P.SCAN", "B.FORCE", "MITM", "DOS", "D.EXFIL"]

# Index 23 is filled from CICIDS-2017 "Flow Bytes/s" (rounded to 10) by the feature mapper,
# despite its contract name; Table 4 must say what the model actually saw.
DISPLAY_NOTE = {23: "Flow Bytes/s (CICIDS-2017) in slot 23"}


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        # raster parts of a figure (the confusion-matrix heat map) at 800 dpi in the PDF: CCPE prefers 800 dpi for graphs
        fig.savefig(_OUT / f"{stem}.{ext}", metadata=_META[ext], dpi=800 if ext == "pdf" else 300)
    plt.close(fig)
    print(f"  [OK] {stem}.pdf / .png")


def _pct(v: float) -> str:
    """Two decimals, never rounding a non-zero error to 0 % or a non-perfect recall to 100 %."""
    if 0 < v < 0.01:
        return "<0.01%"
    if 99.995 <= v < 100:
        return ">99.99%"
    return f"{v:.2f}%"


def fig_confusion(ev: dict) -> None:
    cm = np.array(ev["confusion_matrix"])[:6, :6]
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(pct, cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row percentage (recall on the diagonal)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    for i in range(6):
        for j in range(6):
            n = cm[i, j]
            txt = f"{n:,}\n({_pct(pct[i, j])})" if n else "0"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if pct[i, j] > 50 else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(6), SHORT, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(6), SHORT, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.tight_layout()
    _save(fig, "confusion_matrix_v2")


def fig_per_class(ev: dict) -> None:
    p = [ev["precision_per_class"][c] for c in CLASSES]
    r = [ev["recall_per_class"][c] for c in CLASSES]
    f = [ev["f1_per_class"][c] for c in CLASSES]
    x, w = np.arange(6), 0.25
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for off, vals, lab, col in ((-w, p, "Precision", "#2166ac"), (0, r, "Recall", "#67a9cf"),
                                (w, f, "F1-score", "#d6604d")):
        bars = ax.bar(x + off, vals, w, label=lab, color=col, edgecolor="white", linewidth=0.5)
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.5, rotation=90)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_xticks(x, SHORT, fontsize=7)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.text(2.5, 0.925, "Gate: 0.95", fontsize=6, color="gray", ha="center")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save(fig, "per_class_metrics_v2")


def fig_curves(c: dict) -> None:
    tr, va = c["train_logloss"], c["val_logloss"]
    it = np.arange(1, len(tr) + 1)
    best = c["best_iteration"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    a1.plot(it, tr, lw=0.8, color="#2166ac", label="Train")
    a1.plot(it, va, lw=0.8, color="#d6604d", label="Validation")
    a1.axvline(best, color="gray", ls="--", lw=0.6)
    a1.text(best - 4, max(va) * 0.9, f"Best: {best}", fontsize=6, color="gray", ha="right")
    a1.set_yscale("log")
    a1.set_xlabel("Boosting round")
    a1.set_ylabel("Multi-class log loss (log scale)")
    a1.set_title("(a) Loss")
    a1.legend(framealpha=0.9)
    a1.grid(alpha=0.3, lw=0.5)
    fi = c["f1_iterations"]
    a2.plot(fi, c["train_f1_weighted_subsample200k"], lw=0.8, color="#2166ac", marker=".", ms=3,
            label="Train (200k sample)")
    a2.plot(fi, c["val_f1_weighted"], lw=0.8, color="#d6604d", marker=".", ms=3, label="Validation")
    a2.axvline(best, color="gray", ls="--", lw=0.6)
    a2.axhline(0.95, color="gray", ls=":", lw=0.5)
    a2.text(fi[1], 0.952, "Gate: $F_1^w > 0.95$", fontsize=6, color="gray", ha="left")
    a2.set_xlabel("Boosting round")
    a2.set_ylabel("Weighted F1-score")
    a2.set_title("(b) Weighted F1-score")
    a2.set_ylim(0.94, 1.002)
    a2.legend(loc="center right", framealpha=0.9)
    a2.grid(alpha=0.3, lw=0.5)
    plt.tight_layout()
    _save(fig, "training_curves_v2")


def table4(report: dict) -> None:
    import joblib

    pkl = _MODELS / "lgbm_classifier.pkl"
    digest = hashlib.sha256(pkl.read_bytes()).hexdigest()
    bound = json.loads(_POLICY.read_text())["model"]["sha256"]["lgbm_classifier.pkl"]
    if digest != bound:
        sys.exit(f"lgbm_classifier.pkl SHA-256 {digest} != policy {bound}; refusing")
    booster = joblib.load(pkl).booster_
    split = booster.feature_importance("split")
    gain = booster.feature_importance("gain")
    order = np.argsort(-split)
    rows = [{"rank": k + 1, "index": int(i), "feature": FEATURE_NAMES[i],
             "note": DISPLAY_NOTE.get(int(i), ""), "split": int(split[i]),
             "gain": float(gain[i]), "gain_share": float(gain[i] / gain.sum())}
            for k, i in enumerate(order[:10])]
    unused = [FEATURE_NAMES[i] for i in range(len(split)) if split[i] == 0]
    out = {"model": str(pkl.relative_to(_ROOT)), "sha256": digest,
           "trees": booster.num_trees(), "importance": "split count (gain also given)",
           "top10_by_split": rows,
           "top10_by_gain": [FEATURE_NAMES[i] for i in np.argsort(-gain)[:10]],
           "unused_features": unused, "n_unused": len(unused)}
    (_OUT / "table4_feature_importance_v2.json").write_text(json.dumps(out, indent=2))
    tex = ["% generated by generate_figures_r1.py from " + out["model"] + f" (sha256 {digest[:12]})",
           r"\begin{tabular}{clrr}", r"\toprule",
           r"\textbf{Rank} & \textbf{Feature (index)} & \textbf{Splits} & \textbf{Gain share} \\",
           r"\midrule"]
    for r_ in rows:
        name = r_["feature"].replace("_", r"\_")
        mark = r"$^{a}$" if r_["note"] else ""
        tex.append(f"{r_['rank']} & {name} ({r_['index']}){mark} & {r_['split']:,} & "
                   f"{100 * r_['gain_share']:.1f}\\% \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (_OUT / "table4_feature_importance_v2.tex").write_text("\n".join(tex) + "\n")
    print(f"  [OK] table4_feature_importance_v2.json / .tex ({len(unused)} unused features)")


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    report = json.loads((_MODELS / "training_report.json").read_text())
    ev = report["evaluation"]["test"]
    assert ev["n"] == 364800, ev["n"]
    curves = json.loads((_MODELS / "training_curves_v2.json").read_text())
    print(f"v2 report: trained {report['trained_at']}, test F1w {ev['f1_weighted']:.5f}, "
          f"FPR {100 * ev['false_positive_rate']:.3f} %")
    fig_confusion(ev)
    fig_per_class(ev)
    fig_curves(curves)
    table4(report)


if __name__ == "__main__":
    main()
