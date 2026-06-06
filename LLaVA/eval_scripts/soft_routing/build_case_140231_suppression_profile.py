#!/usr/bin/env python3
"""Build suppression-profile figure for qualitative case 140231."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
GREEN = "#059669"
RED = "#dc2626"
BLUE = "#2563eb"
ORANGE = "#f97316"


def read_rows(path: Path, question_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("question_id") == question_id]
    if not rows:
        raise SystemExit(f"no rows for question_id={question_id} in {path}")
    rows.sort(key=lambda row: int(row["step_idx"]))
    return rows


def ff(row: dict[str, str], key: str) -> float:
    return float(row[key])


def ii(row: dict[str, str], key: str) -> int:
    return int(row[key])


def find_token(rows: list[dict[str, str]], token: str, label: str) -> dict[str, str]:
    for row in rows:
        if row["token_text"].strip().lower() == token and row["label"] == label:
            return row
    raise SystemExit(f"missing {label} token={token}")


def display_token(token: str) -> str:
    token = token.strip()
    replacements = {
        "ones": "cell phone",
        "books": "book",
    }
    return replacements.get(token.lower(), token)


def add_summary_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, token: str, delta: float, ratio: float, outcome: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#ffffff",
            alpha=0.98,
            clip_on=False,
        )
    )
    ax.text(x + 0.035, y + h - 0.075, title, transform=ax.transAxes, fontsize=10.5, weight="bold", color=color, va="center")
    ax.text(x + 0.035, y + h - 0.185, token, transform=ax.transAxes, fontsize=14.0, weight="bold", color=DARK, va="center")
    ax.text(x + 0.035, y + 0.096, rf"$\delta$={delta:.3f}   $r$={ratio:.3f}", transform=ax.transAxes, fontsize=9.6, weight="bold", color=MUTED, va="center")
    ax.text(x + 0.035, y + 0.04, outcome, transform=ax.transAxes, fontsize=9.9, weight="bold", color=color, va="center")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-csv",
        default="LLaVA/results/coco/dual_ratio_detector_diagnostics_top100_l9_l16/dual_ratio_token_rows.csv",
    )
    parser.add_argument("--question-id", default="140231")
    parser.add_argument("--output-dir", default="LLaVA/results/coco/qualitative_case_studies/figures")
    args = parser.parse_args()

    rows = read_rows(Path(args.token_csv), args.question_id)
    keyboard = find_token(rows, "keyboard", "grounded")
    person = find_token(rows, "person", "hallucinated")

    x = np.arange(len(rows))
    tokens = [display_token(row["token_text"]) for row in rows]
    steps = [ii(row, "step_idx") for row in rows]
    ratios = np.array([ff(row, "r_img") for row in rows])
    deltas = np.array([ff(row, "delta_img") for row in rows])
    labels = [row["label"] for row in rows]
    colors = [RED if label == "hallucinated" else GREEN for label in labels]

    fig = plt.figure(figsize=(7.8, 4.35), dpi=260)
    fig.patch.set_facecolor("#f8fafc")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.74, 1.0], hspace=0.18)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.set_axis_off()
    ax_top.text(0.0, 0.98, "Case 140231: grounded survives under comparable suppression", fontsize=16.3, weight="bold", color=DARK, va="top")
    ax_top.text(
        0.0,
        0.78,
        "Greedy-prefix probes show similar dynamic suppression; the final caption removes the hallucinated object but preserves the grounded keyboard.",
        fontsize=10.7,
        weight="semibold",
        color=MUTED,
        va="top",
    )
    add_summary_box(
        ax_top,
        0.02,
        0.05,
        0.45,
        0.56,
        "hallucinated object",
        "person  ->  none",
        ff(person, "delta_img"),
        ff(person, "r_img"),
        "caption outcome: erased",
        RED,
    )
    add_summary_box(
        ax_top,
        0.53,
        0.05,
        0.45,
        0.56,
        "grounded object",
        "keyboard  ->  keyboard",
        ff(keyboard, "delta_img"),
        ff(keyboard, "r_img"),
        "caption outcome: preserved",
        GREEN,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#ffffff")
    ax.bar(x, deltas, width=0.64, color=colors, alpha=0.22, edgecolor=colors, linewidth=1.4)
    ax.plot(x, deltas, color="#334155", lw=1.2, alpha=0.45, zorder=2)
    ax.scatter(x, deltas, s=36, color=colors, edgecolor="white", linewidth=0.9, zorder=3)

    for idx, row in enumerate(rows):
        token = row["token_text"].strip().lower()
        if token == "keyboard" and row["label"] == "grounded":
            ax.scatter([idx], [ff(row, "delta_img")], s=165, facecolors="none", edgecolors=GREEN, linewidths=2.8, zorder=5)
            ax.text(idx, ff(row, "delta_img") + 0.075, "keyboard\npreserved", ha="center", va="bottom", fontsize=9.2, weight="bold", color=GREEN)
        if token == "person" and row["label"] == "hallucinated":
            ax.scatter([idx], [ff(row, "delta_img")], s=165, facecolors="none", edgecolors=RED, linewidths=2.8, zorder=5)
            ax.text(idx, ff(row, "delta_img") + 0.075, "person\nerased", ha="center", va="bottom", fontsize=9.2, weight="bold", color=RED)

    ax2 = ax.twinx()
    ax2.plot(x, ratios, color=BLUE, lw=2.0, alpha=0.82, marker="o", ms=3.2, label=r"$r=T/(T+I)$")
    ax2.axhline(0.90, color=ORANGE, lw=1.25, ls=(0, (4, 3)), alpha=0.85)
    ax2.text(len(rows) - 0.1, 0.905, r"$\tau=0.90$", ha="right", va="bottom", fontsize=8.5, weight="bold", color=ORANGE)

    ax.set_ylabel(r"suppression $\delta$", fontsize=10.4, weight="bold", color=DARK)
    ax2.set_ylabel(r"text reliance $r$", fontsize=10.4, weight="bold", color=BLUE)
    ax.set_xlabel("greedy object-token order", fontsize=10.4, weight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{tok}\n@{step}" for tok, step in zip(tokens, steps)], fontsize=7.4, rotation=0)
    ax.set_ylim(0, max(0.95, float(deltas.max()) + 0.16))
    ax2.set_ylim(0.50, 0.98)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.tick_params(axis="y", labelsize=8.2, colors="#334155")
    ax2.tick_params(axis="y", labelsize=8.2, colors=BLUE)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["top", "left"]:
        ax2.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax2.spines["right"].set_color("#cbd5e1")

    fig.subplots_adjust(left=0.085, right=0.91, bottom=0.14, top=0.93)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "case_140231_person_keyboard_suppression_profile"
    for ext in ["png", "svg", "pdf"]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
