#!/usr/bin/env python3
"""Build a compact top-5 next-token shift figure from a case logit probe CSV."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
HALL = "#e11d48"
GROUND = "#059669"
BASE = "#f97316"
DEACT = "#2563eb"
TARGET_BG = "#fee2e2"
BAR_BG = "#ffedd5"
PAPER = "#f8fafc"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty probe CSV: {path}")
    return rows


def safe_float(value: str | None, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_token(token: str | None) -> str:
    token = str(token or "").replace("\\n", " ").replace("\n", " ").strip()
    token = token.replace("▁", " ").replace("Ġ", " ").strip()
    return token if token else "<space>"


def candidates(row: dict[str, str], mode: str, top_k: int, score_mode: str) -> list[dict[str, object]]:
    target_id = str(row.get("target_token_id", ""))
    out: list[dict[str, object]] = []
    for rank in range(1, top_k + 1):
        token_key = f"{mode}_top{rank}_token"
        if token_key not in row or row.get(token_key, "") == "":
            continue
        token_id = str(row.get(f"{mode}_top{rank}_id", ""))
        out.append(
            {
                "rank": rank,
                "token": clean_token(row.get(token_key)),
                "score": safe_float(row.get(f"{mode}_top{rank}_{score_mode}")),
                "is_target": token_id == target_id,
            }
        )
    if not any(item["is_target"] for item in out):
        target_score = safe_float(row.get(f"{mode}_target_{score_mode}"))
        target_rank = int(safe_float(row.get(f"{mode}_target_rank"), 0))
        if not math.isnan(target_score) and target_rank > 0:
            out.append(
                {
                    "rank": target_rank,
                    "token": clean_token(row.get("target_token")),
                    "score": target_score,
                    "is_target": True,
                    "extra": True,
                }
            )
    return out


def score_label(score_mode: str) -> str:
    return {"prob": "probability", "logprob": "log probability", "logit": "logit"}[score_mode]


def format_score(value: float, score_mode: str) -> str:
    if math.isnan(value):
        return ""
    if score_mode == "prob":
        return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"
    return f"{value:.2f}"


def draw_candidate_axis(ax: plt.Axes, items: list[dict[str, object]], color: str, score_mode: str, xlim: float) -> None:
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="x", labelsize=7.2, colors=MUTED)
    ax.tick_params(axis="y", length=0, labelsize=9.5, colors=DARK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0.0, xlim)

    labels = [f"#{item['rank']}" for item in items]
    values = [float(item["score"]) for item in items]
    ypos = list(range(len(items)))[::-1]
    colors = [color if item["is_target"] else BAR_BG for item in items]
    edges = [HALL if item["is_target"] else "#fed7aa" for item in items]
    ax.barh(ypos, values, color=colors, edgecolor=edges, height=0.62, linewidth=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)

    for y, value, item in zip(ypos, values, items):
        ax.text(
            min(value + xlim * 0.025, xlim * 0.98),
            y,
            format_score(value, score_mode),
            ha="left",
            va="center",
            fontsize=8.2,
            color=HALL if item["is_target"] else MUTED,
            weight="bold" if item["is_target"] else "semibold",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-mode", choices=["prob", "logprob", "logit"], default="prob")
    parser.add_argument("--formats", default="png,svg,pdf")
    args = parser.parse_args()

    rows = read_rows(Path(args.probe_csv))
    rows.sort(key=lambda row: (0 if row.get("label") == "hallucinated" else 1, row.get("object_word", "")))
    image_id = rows[0].get("image_id", "case")

    fig = plt.figure(figsize=(7.25, 1.15 + 1.55 * len(rows)), dpi=260)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(
        nrows=len(rows) + 1,
        ncols=5,
        height_ratios=[0.28] + [1.0] * len(rows),
        width_ratios=[1.08, 2.0, 0.58, 2.0, 0.03],
        hspace=0.34,
        wspace=0.18,
        left=0.055,
        right=0.985,
        top=0.93,
        bottom=0.075,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.55,
        "Top-5 next-token shift under DEACT",
        ha="left",
        va="center",
        fontsize=14.3,
        weight="bold",
        color=DARK,
    )
    ax_title.text(0.99, 0.55, f"image {image_id}", ha="right", va="center", fontsize=8.6, weight="bold", color=MUTED)

    for row_idx, row in enumerate(rows, start=1):
        label = row.get("label", "")
        object_word = row.get("object_word", "")
        object_node = row.get("object_node", object_word)
        target_token = clean_token(row.get("target_token"))
        color = HALL if label == "hallucinated" else GROUND
        base_items = candidates(row, "base", args.top_k, args.score_mode)
        deact_items = candidates(row, "deact", args.top_k, args.score_mode)
        max_value = max([float(item["score"]) for item in base_items + deact_items] + [1e-6])
        xlim = max_value * 1.28 if args.score_mode == "prob" else max_value + abs(max_value) * 0.18 + 0.25

        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.axis("off")
        ax_label.text(0.0, 0.66, "H" if label == "hallucinated" else "G", ha="left", va="center", fontsize=10.5, weight="bold", color=color)
        ax_label.text(0.16, 0.66, object_word, ha="left", va="center", fontsize=13.2, weight="bold", color=DARK)
        ax_label.text(0.16, 0.39, target_token, ha="left", va="center", fontsize=7.7, color=MUTED, weight="semibold")

        ax_base = fig.add_subplot(gs[row_idx, 1])
        ax_deact = fig.add_subplot(gs[row_idx, 3])
        draw_candidate_axis(ax_base, base_items, color, args.score_mode, xlim)
        draw_candidate_axis(ax_deact, deact_items, color, args.score_mode, xlim)
        ax_base.set_title("Base", fontsize=9.4, weight="bold", color=BASE, pad=4)
        ax_deact.set_title("DEACT", fontsize=9.4, weight="bold", color=DEACT, pad=4)

        ax_mid = fig.add_subplot(gs[row_idx, 2])
        ax_mid.axis("off")
        drop = safe_float(row.get("target_logprob_drop"))
        base_rank = row.get("base_target_rank", "")
        deact_rank = row.get("deact_target_rank", "")
        ax_mid.add_patch(
            FancyArrowPatch(
                (0.08, 0.52),
                (0.92, 0.52),
                transform=ax_mid.transAxes,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.0,
                color=color,
                alpha=0.88,
            )
        )
        ax_mid.text(0.5, 0.75, f"{drop:.3f}", ha="center", va="center", fontsize=9.0, weight="bold", color=color)
        ax_mid.text(0.5, 0.25, f"{base_rank}->{deact_rank}", ha="center", va="center", fontsize=8.3, weight="bold", color=DARK)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"case_{image_id}_top5_candidate_shift_numeric"
    for ext in [item.strip() for item in args.formats.split(",") if item.strip()]:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
