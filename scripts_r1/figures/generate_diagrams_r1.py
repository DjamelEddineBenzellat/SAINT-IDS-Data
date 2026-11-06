"""
Figures 1-2 for manuscript R1: architecture and per-flow pipeline, drawn from code so they can be
edited and re-checked. They show only what is implemented in the deployed path (policy 2.4.0):

  Fig. 1  edge node -> untrusted receiver (public key only) -> hospital key holder (secret key)
  Fig. 2  ingest -> score + decide -> explain -> route, with the asynchronous encrypted relay;
          stage times are the in-process means of Table 8 (comment13/stage_r1_report.md)

Writes CCPE_revision1/figures/figure1_r1.{pdf,png} and figure2_r1.{pdf,png}.

Usage: python realtime-ids-healthcare-paper/CCPE_revision1/generate_diagrams_r1.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_OUT = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
})

# No creation dates or software tags, so re-running gives byte-identical files.
_META = {"pdf": {"CreationDate": None, "Producer": None, "Creator": None}, "png": {"Software": None}}

INK = "#1b2a41"
BLUE = "#2166ac"
GREEN = "#1b7837"
ORANGE = "#b35806"
GREY = "#6b7280"
FILL = {"edge": "#eef4fb", "cloud": "#fdf3e7", "key": "#edf7ef", "box": "white"}


def box(ax, x, y, w, h, title, lines=(), edge=INK, fill="white", title_size=8.5, size=7, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.058 - k * 0.032, line, ha="center", va="top", fontsize=size,
                color="#374151")


def region(ax, x, y, w, h, label, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc=fill, ec=edge, lw=0.9, ls=(0, (4, 2))))
    ax.text(x + 0.012, y + h - 0.012, label, ha="left", va="top", fontsize=8, fontweight="bold",
            color=edge)


def arrow(ax, p, q, text="", color=INK, ls="-", text_offset=(0, 0.018), size=6.5, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.0,
                                 linestyle=ls, shrinkA=1, shrinkB=1))
    if text:
        mx, my = (p[0] + q[0]) / 2 + text_offset[0], (p[1] + q[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha=ha, va="bottom", fontsize=size, color=color, style="italic")


def figure1() -> None:
    fig = plt.figure(figsize=(7.16, 3.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.525)
    ax.axis("off")

    # Edge node (hospital network)
    region(ax, 0.005, 0.205, 0.99, 0.315, "EDGE NODE (hospital network, x86 gateway)", FILL["edge"], BLUE)
    yb, hb = 0.27, 0.18
    box(ax, 0.02, yb, 0.175, hb, "Flow collector",
        ["headers only, no payload", "33-feature flow vector", "DICOM 104/11112, HL7 2575,", "FHIR 443/8080 (port flags)"])
    box(ax, 0.225, yb, 0.2, hb, "Scoring and decision",
        ["LightGBM (ONNX Runtime)", r"alert if $s=1-p_{\mathrm{BENIGN}}\geq\tau$", r"$\tau=0.25$; + fast filter", "evasion rules: signals only"])
    box(ax, 0.455, yb, 0.165, hb, "Explanation",
        ["TreeSHAP (exact,", "path-dependent)", "top-5 features,", "every flow"])
    box(ax, 0.65, yb, 0.155, hb, "Alert router",
        ["API response", "local log: keyed daily", "IP pseudonyms (HMAC)", "raw IPs only in memory"])
    box(ax, 0.83, yb, 0.155, hb, "Encrypted relay",
        ["alerts only, async", "30 values per alert", "CKKS, public key", r"$\leq$136 per ciphertext"],
        edge=ORANGE)
    ym = yb + hb / 2
    for x0, x1 in ((0.195, 0.225), (0.425, 0.455), (0.62, 0.65), (0.805, 0.83)):
        arrow(ax, (x0, ym), (x1, ym))
    ax.text(0.40, 0.213, "Leaves the edge: only CKKS ciphertexts of alerts (no IP address, port,\n"
            "time-of-day or payload). Batch time (to 250 ms) and batch size are visible.",
            ha="center", va="bottom", fontsize=7, color=GREY, linespacing=1.3)
    ax.text(0.895, 0.222, "packed batches,\nflushed when full\nor after 250 ms", ha="right", va="bottom",
            fontsize=6.3, color=ORANGE, style="italic", linespacing=1.15)

    # Key holder (middle) and receiver (right, under the relay)
    region(ax, 0.30, 0.005, 0.32, 0.185, "HOSPITAL KEY HOLDER (SOC)", FILL["key"], GREEN)
    box(ax, 0.315, 0.018, 0.29, 0.135, "Decryption",
        ["secret key", "batches for incident response", "aggregates (minimum window)"], edge=GREEN)
    region(ax, 0.65, 0.005, 0.345, 0.185, "UNTRUSTED RECEIVER (cloud)", FILL["cloud"], ORANGE)
    box(ax, 0.665, 0.018, 0.315, 0.135, "Encrypted alert receiver",
        ["verifies HMAC and key id", "homomorphic aggregation", "public key only: cannot decrypt"],
        edge=ORANGE)

    arrow(ax, (0.9075, yb), (0.9075, 0.155), color=ORANGE)
    arrow(ax, (0.665, 0.0855), (0.605, 0.0855), color=GREEN)
    ax.text(0.15, 0.095, "Encryption parameters fixed in the\npolicy file: N = 8192, moduli\n"
            "[60, 40, 40, 60] bits, scale $2^{40}$", ha="center", va="center", fontsize=6.8, color=GREY)

    for ext in ("pdf", "png"):
        fig.savefig(_OUT / f"figure1_r1.{ext}", metadata=_META[ext])
    plt.close(fig)
    print("  [OK] figure1_r1.pdf / .png")


def figure2() -> None:
    fig = plt.figure(figsize=(7.16, 2.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.065, 0.355)
    ax.axis("off")

    stages = [
        ("1  Ingest", ["validate request", "33-feature vector", "NaN / inf cleaned", "(no z-scaling)"], "0.05 ms", BLUE),
        ("2  Score and decide", ["ONNX LightGBM", r"$s=1-p_{\mathrm{BENIGN}}$", r"alert if $s\geq0.25$", "or fast-filter rule"], "0.20 ms", BLUE),
        ("3  Explain", ["native TreeSHAP", "top-5 features", "status ok / failed", "(never dropped)"], "8.2 ms", BLUE),
        ("4  Route", ["response + log", "IP pseudonyms", "alert sources,", "policy version"], "0.12 ms", BLUE),
    ]
    w, h, y, gap, x = 0.16, 0.205, 0.105, 0.022, 0.11
    ax.text(0.055, y + h / 2, "Flow\nmetadata", ha="center", va="center", fontsize=8,
            fontweight="bold", color=INK)
    arrow(ax, (0.09, y + h / 2), (x, y + h / 2))
    for k, (title, lines, t, col) in enumerate(stages):
        xs = x + k * (w + gap)
        box(ax, xs, y, w, h, title, lines, edge=col)
        ax.text(xs + w / 2, y - 0.012, t, ha="center", va="top", fontsize=7.5, color=col,
                fontweight="bold")
        if k < len(stages) - 1:
            arrow(ax, (xs + w, y + h / 2), (xs + w + gap, y + h / 2))
    xr = x + 4 * (w + gap) - gap
    box(ax, 0.845, y, 0.145, h, "Encrypted relay", ["alerts only, queued", "on a background thread", "CKKS + HMAC", "4.3 ms per ciphertext"],
        edge=ORANGE)
    arrow(ax, (xr, y + h / 2), (0.845, y + h / 2), color=ORANGE, ls=(0, (3, 2)))
    ax.text(0.9175, y - 0.012, r"delivery p99 264 ms", ha="center", va="top", fontsize=7.5,
            color=ORANGE, fontweight="bold")
    ax.annotate("", xy=(x, 0.335), xytext=(xr, 0.335),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text((x + xr) / 2, 0.338, "request path: 8.6 ms mean, 13.7 ms p99 per flow (one core, in-process)",
            ha="center", va="bottom", fontsize=7, color=GREY)

    for ext in ("pdf", "png"):
        fig.savefig(_OUT / f"figure2_r1.{ext}", metadata=_META[ext])
    plt.close(fig)
    print("  [OK] figure2_r1.pdf / .png")


if __name__ == "__main__":
    _OUT.mkdir(exist_ok=True)
    figure1()
    figure2()
