#!/usr/bin/env python3
"""Visualize base vs DEACT next-token candidates for qualitative object probes."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
GREEN = "#059669"
RED = "#dc2626"
BLUE = "#2563eb"
ORANGE = "#fb923c"
CYAN = "#dff7fb"
PAPER = "#f8fafc"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    return rows


def safe_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def clean_token(token: str) -> str:
    token = str(token).replace("\\n", " ").replace("\n", " ").strip()
    token = token.replace("▁", " ").replace("Ġ", " ").strip()
    return token or "<space>"


def mode_candidates(row: dict[str, str], mode: str, top_k: int, score_mode: str) -> list[dict[str, object]]:
    candidates = []
    for rank in range(1, top_k + 1):
        token_key = f"{mode}_top{rank}_token"
        if token_key not in row or row.get(token_key, "") == "":
            continue
        token = clean_token(row[token_key])
        score = safe_float(row, f"{mode}_top{rank}_{score_mode}")
        candidates.append(
            {
                "rank": rank,
                "token": token,
                "score": score,
                "token_id": row.get(f"{mode}_top{rank}_id", ""),
                "is_target": str(row.get(f"{mode}_top{rank}_id", "")) == str(row.get("target_token_id", "")),
            }
        )
    target_token = clean_token(row.get("target_token", ""))
    target_rank_key = f"{mode}_target_rank"
    target_rank = int(float(row.get(target_rank_key, "0") or 0))
    if not any(item["is_target"] for item in candidates):
        target_score = safe_float(row, f"{mode}_target_{score_mode}")
        if not math.isnan(target_score):
            candidates.append(
                {
                    "rank": target_rank,
                    "token": target_token,
                    "score": target_score,
                    "token_id": row.get("target_token_id", ""),
                    "is_target": True,
                    "is_extra_target": True,
                }
            )
    return candidates


def display_score(value: float, score_mode: str) -> str:
    if math.isnan(value):
        return ""
    if score_mode == "prob":
        return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"
    if score_mode == "logprob":
        return f"{value:.2f}"
    return f"{value:.2f}"


def candidate_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    candidates: list[dict[str, object]],
    color: str,
    score_mode: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#ffffff",
            alpha=0.98,
        )
    )
    ax.text(x + w / 2, y + h - 0.048, title, ha="center", va="center", fontsize=10.8, weight="bold", color=color)
    n = max(len(candidates), 1)
    row_h = (h - 0.1) / n
    for idx, item in enumerate(candidates):
        yy = y + h - 0.1 - (idx + 1) * row_h
        is_target = bool(item.get("is_target"))
        fill = "#fff7ed" if not is_target else "#fee2e2"
        edge = "#fed7aa" if not is_target else RED
        if idx % 2 == 0 and not is_target:
            fill = "#ffedd5"
        ax.add_patch(Rectangle((x + 0.01, yy + 0.006), w - 0.02, row_h - 0.012, facecolor=fill, edgecolor=edge, linewidth=1.0 if is_target else 0.45))
        rank = item.get("rank")
        token = str(item.get("token", ""))
        score = display_score(float(item.get("score", float("nan"))), score_mode)
        label_color = RED if is_target else DARK
        weight = "bold" if is_target else "semibold"
        prefix = f"#{rank} " if rank else ""
        if item.get("is_extra_target"):
            prefix = f"target r{rank} "
        ax.text(x + 0.025, yy + row_h / 2, f"{prefix}{token}", ha="left", va="center", fontsize=8.2, weight=weight, color=label_color)
        ax.text(x + w - 0.025, yy + row_h / 2, score, ha="right", va="center", fontsize=8.1, weight=weight, color=label_color)


def draw_case_row(ax: plt.Axes, row: dict[str, str], y: float, h: float, top_k: int, score_mode: str) -> None:
    label = row.get("label", "")
    obj = row.get("object_word", "")
    is_hall = label == "hallucinated"
    color = RED if is_hall else GREEN
    outcome = "final caption: absent" if is_hall else "final caption: present"
    transition = f"{obj}"

    ax.add_patch(
        FancyBboxPatch(
            (0.025, y),
            0.95,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.022",
            linewidth=1.0,
            edgecolor="#e2e8f0",
            facecolor="#ffffff",
            alpha=0.88,
        )
    )
    ax.text(0.045, y + h - 0.055, f"{label} object", fontsize=9.8, weight="bold", color=color, ha="left", va="center")
    ax.text(0.045, y + h - 0.118, transition, fontsize=16.0, weight="bold", color=DARK, ha="left", va="center")
    ax.text(0.045, y + 0.045, outcome, fontsize=8.8, weight="bold", color=color, ha="left", va="center")

    base = mode_candidates(row, "base", top_k, score_mode)
    deact = mode_candidates(row, "deact", top_k, score_mode)

    box_y = y + 0.045
    box_h = h - 0.09
    candidate_box(ax, 0.255, box_y, 0.265, box_h, "Base", base, ORANGE, score_mode)
    candidate_box(ax, 0.705, box_y, 0.265, box_h, "DEACT", deact, BLUE, score_mode)
    ax.add_patch(
        FancyArrowPatch(
            (0.545, y + h / 2),
            (0.68, y + h / 2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.2,
            color=color,
            alpha=0.9,
        )
    )
    drop = safe_float(row, "target_logprob_drop")
    rank_base = row.get("base_target_rank", "")
    rank_deact = row.get("deact_target_rank", "")
    top1_base = clean_token(row.get("base_top1_token", ""))
    top1_deact = clean_token(row.get("deact_top1_token", ""))
    ax.text(
        0.612,
        y + h / 2 + 0.067,
        rf"$\Delta\log p$={drop:.3f}" if not math.isnan(drop) else "",
        fontsize=8.8,
        weight="bold",
        color=color,
        ha="center",
        va="center",
    )
    ax.text(
        0.612,
        y + h / 2 - 0.06,
        f"target rank {rank_base}->{rank_deact}\ntop1: {top1_base} -> {top1_deact}",
        fontsize=7.1,
        color=MUTED,
        ha="center",
        va="center",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-csv",
        default="LLaVA/results/coco/qualitative_case_studies/logit_probe_140231/case_object_logit_probe_rows.csv",
    )
    parser.add_argument("--output-dir", default="LLaVA/results/coco/qualitative_case_studies/figures")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-mode", choices=["prob", "logprob", "logit"], default="prob")
    parser.add_argument("--formats", default="png,svg,pdf")
    args = parser.parse_args()

    rows = read_rows(Path(args.probe_csv))
    rows.sort(key=lambda row: 0 if row.get("label") == "hallucinated" else 1)

    fig_h = 2.3 + 2.15 * len(rows)
    fig = plt.figure(figsize=(8.3, fig_h), dpi=260)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.04, 0.96, "Greedy-prefix next-token candidates under DEACT", fontsize=16.4, weight="bold", color=DARK, ha="left", va="top")
    ax.text(
        0.04,
        0.918,
        "This is a local one-step probe at the greedy object-token prefix; final caption outcome is shown separately.",
        fontsize=9.3,
        weight="semibold",
        color=MUTED,
        ha="left",
        va="top",
    )
    score_label = {"prob": "softmax probability", "logprob": "log probability", "logit": "logit"}[args.score_mode]
    ax.text(0.96, 0.918, f"values: {score_label}", fontsize=8.2, color=MUTED, ha="right", va="top")

    top_margin = 0.12
    bottom_margin = 0.04
    usable = 1.0 - top_margin - bottom_margin
    row_h = usable / len(rows)
    for idx, row in enumerate(rows):
        y = bottom_margin + (len(rows) - 1 - idx) * row_h + 0.02
        draw_case_row(ax, row, y, row_h - 0.035, args.top_k, args.score_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "case_140231_logit_candidate_flow"
    for ext in [item.strip() for item in args.formats.split(",") if item.strip()]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
