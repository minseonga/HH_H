#!/usr/bin/env python3
"""Build the layer-wise fragility selectivity figure for Section III-E."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROWS = [
    {
        "band": "L0-8",
        "grounded_delta": -0.003879136630898251,
        "hall_delta": -0.0023612980843366435,
        "gap": 0.0015178385465616075,
        "grounded_flip": 0.008888888888888889,
        "hall_flip": 0.007777777777777778,
    },
    {
        "band": "L9-16",
        "grounded_delta": 0.0018952798875807278,
        "hall_delta": 0.020733409860404207,
        "gap": 0.018838129972823477,
        "grounded_flip": 0.0325,
        "hall_flip": 0.065,
    },
    {
        "band": "L17-24",
        "grounded_delta": 0.009989882261384083,
        "hall_delta": 0.02453878583626647,
        "gap": 0.014548903574882388,
        "grounded_flip": 0.051250000000000004,
        "hall_flip": 0.05375,
    },
    {
        "band": "L25-31",
        "grounded_delta": 0.01596088312865634,
        "hall_delta": 0.028373412930606197,
        "gap": 0.01241252980194986,
        "grounded_flip": 0.03428571428571429,
        "hall_flip": 0.06571428571428571,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="LLaVA/results/coco/layerwise_fragility_selectivity_figure")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = DEFAULT_ROWS
    with (out_dir / "layerwise_fragility_selectivity_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "band",
                "grounded_delta",
                "hall_delta",
                "gap",
                "selectivity_hall_over_ground",
                "grounded_flip",
                "hall_flip",
                "flip_gap",
            ],
        )
        writer.writeheader()
        for row in rows:
            grounded = row["grounded_delta"]
            hall = row["hall_delta"]
            selectivity = hall / grounded if grounded > 0 else ""
            writer.writerow(
                {
                    **row,
                    "selectivity_hall_over_ground": selectivity,
                    "flip_gap": row["hall_flip"] - row["grounded_flip"],
                }
            )

    bands = [row["band"] for row in rows]
    x = np.arange(len(rows))
    grounded = np.array([row["grounded_delta"] for row in rows], dtype=float)
    hall = np.array([row["hall_delta"] for row in rows], dtype=float)
    gap = np.array([row["gap"] for row in rows], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.85, 3.75), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFF")
    ax.axhline(0, color="#94A3B8", lw=1.1, zorder=1)
    ax.grid(axis="y", color="#E5E7EB", lw=0.85, zorder=0)

    hall_bars = ax.bar(
        x,
        hall,
        width=0.58,
        color="#FFEDD5",
        edgecolor="#F97316",
        alpha=0.72,
        linewidth=1.8,
        label="hallucinated",
        zorder=2,
    )
    grounded_bars = ax.bar(
        x,
        grounded,
        width=0.58,
        color="#DCFCE7",
        edgecolor="#16A34A",
        alpha=0.72,
        linewidth=1.8,
        label="grounded",
        zorder=3,
    )
    (gap_line,) = ax.plot(
        x,
        gap,
        color="#7C3AED",
        lw=2.8,
        marker="o",
        ms=7.8,
        mfc="#F3E8FF",
        mec="#7C3AED",
        mew=2.0,
        label="H-G gap",
        zorder=5,
    )

    # Small value labels make the mixed bar/line chart self-contained without a separate table.
    for xi, g_val, h_val, gap_val in zip(x, grounded, hall, gap):
        h_va = "bottom" if h_val >= 0 else "top"
        h_offset = 0.0011 if h_val >= 0 else -0.0011
        g_va = "top" if g_val >= 0 else "bottom"
        g_offset = -0.0011 if g_val >= 0 else 0.0011
        ax.text(xi + 0.18, h_val + h_offset, f"{h_val:.3f}", ha="left", va=h_va, fontsize=7.5, color="#9A3412", weight="bold")
        ax.text(xi - 0.18, g_val + g_offset, f"{g_val:.3f}", ha="right", va=g_va, fontsize=7.5, color="#166534", weight="bold")
        ax.text(xi + 0.05, gap_val + 0.0007, f"{gap_val:.3f}", ha="left", va="bottom", fontsize=7.3, color="#5B21B6", weight="bold")

    ax.set_title("Layer-wise text-side actuation", fontsize=15.5, weight="bold", color="#111827", pad=8)
    ax.set_ylabel(r"object-token $\Delta \log p$", fontsize=10.5, color="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=9.4, fontweight="bold", color="#111827")
    ax.tick_params(axis="y", labelsize=8.5, colors="#475569")
    ax.set_ylim(-0.006, 0.0335)
    ax.set_xlim(-0.38, len(rows) - 0.62)
    ax.legend(
        handles=[grounded_bars, hall_bars, gap_line],
        labels=["grounded", "hallucinated", "H-G gap"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        fontsize=8.3,
        handlelength=2.0,
        columnspacing=1.2,
    )
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"layerwise_fragility_line_plot.{fmt}", bbox_inches="tight")
    plt.close(fig)

    # Cleaner manuscript variant: show absolute drops as overlapped bars and selectivity as one line.
    fig, ax = plt.subplots(figsize=(4.05, 3.12), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFF")
    ax.axhline(0, color="#94A3B8", lw=1.0, zorder=1)
    ax.grid(axis="y", color="#E5E7EB", lw=0.75, zorder=0)

    hall_bars = ax.bar(
        x,
        hall,
        width=0.58,
        color="#FFEDD5",
        edgecolor="#F97316",
        alpha=0.72,
        linewidth=1.45,
        label="hallucinated",
        zorder=2,
    )
    grounded_bars = ax.bar(
        x,
        grounded,
        width=0.58,
        color="#DCFCE7",
        edgecolor="#16A34A",
        alpha=0.72,
        linewidth=1.45,
        label="grounded",
        zorder=3,
    )
    (gap_line,) = ax.plot(
        x,
        gap,
        color="#7C3AED",
        lw=2.45,
        marker="o",
        ms=6.7,
        mfc="#F3E8FF",
        mec="#7C3AED",
        mew=1.7,
        label="H-G gap",
        zorder=5,
    )

    ax.set_title("Layer-wise Text-Side Actuation", fontsize=11.4, weight="bold", color="#111827", pad=7)
    ax.set_ylabel(r"$\Delta \log p$ at object token", fontsize=8.7, color="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=8.4, fontweight="bold", color="#111827")
    ax.tick_params(axis="y", labelsize=7.7, colors="#475569")
    ax.set_ylim(-0.006, 0.0325)
    ax.set_xlim(-0.35, len(rows) - 0.65)
    ax.legend(
        handles=[grounded_bars, hall_bars, gap_line],
        labels=["grounded", "hallucinated", "H-G gap"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=3,
        fontsize=7.4,
        handlelength=1.8,
        columnspacing=0.9,
    )
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"layerwise_fragility_line_plot_clean.{fmt}", bbox_inches="tight")
    plt.close(fig)

    width, height = 900, 560
    left, right, top, bottom = 95, 60, 72, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = -0.006, 0.0185
    y_min, y_max = -0.005, 0.033

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    colors = {
        "L0-8": "#a8b4c5",
        "L9-16": "#7c3aed",
        "L17-24": "#f97316",
        "L25-31": "#ef4444",
    }
    label_pos = {
        "L0-8": (-0.0032, 0.004),
        "L9-16": (0.0048, 0.0177),
        "L17-24": (0.011, 0.0205),
        "L25-31": (0.014, 0.0305),
    }
    label_text = {
        "L0-8": ["L0-8", "near zero"],
        "L9-16": ["L9-16", "large H-G gap, low G"],
        "L17-24": ["L17-24", "stronger collateral"],
        "L25-31": ["L25-31", "late collateral"],
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcff" stroke="#d7dee8" stroke-width="1"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Layer-wise intervention-response selectivity</text>',
        f'<text x="{width/2}" y="56" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#64748b">x: grounded object log-probability drop; y: hallucinated object log-probability drop</text>',
    ]

    for x in [-0.005, 0.0, 0.005, 0.010, 0.015]:
        px = sx(x)
        svg.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top+plot_h}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{px:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">{x:.3f}</text>')
    for y in [0.0, 0.01, 0.02, 0.03]:
        py = sy(y)
        svg.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left+plot_w}" y2="{py:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{left-14}" y="{py+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#475569">{y:.2f}</text>')

    svg.append(f'<line x1="{sx(0):.2f}" y1="{top}" x2="{sx(0):.2f}" y2="{top+plot_h}" stroke="#94a3b8" stroke-width="1.2"/>')
    svg.append(f'<line x1="{left}" y1="{sy(0):.2f}" x2="{left+plot_w}" y2="{sy(0):.2f}" stroke="#94a3b8" stroke-width="1.2"/>')
    svg.append(f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">Grounded object Δ log p</text>')
    svg.append(f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">Hallucinated object Δ log p</text>')

    box_x, box_y = sx(0.0005), sy(0.0305)
    svg.append(f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="205" height="61" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>')
    svg.append(f'<text x="{box_x+12:.2f}" y="{box_y+21:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">desired region:</text>')
    svg.append(f'<text x="{box_x+12:.2f}" y="{box_y+39:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">high hall fragility</text>')
    svg.append(f'<text x="{box_x+12:.2f}" y="{box_y+57:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">low grounded collateral</text>')

    for row in rows:
        band = row["band"]
        x = row["grounded_delta"]
        y = row["hall_delta"]
        px, py = sx(x), sy(y)
        lx, ly = sx(label_pos[band][0]), sy(label_pos[band][1])
        radius = 13 if band == "L9-16" else 10
        svg.append(f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{lx-7:.2f}" y2="{ly:.2f}" stroke="#6b7280" stroke-width="1"/>')
        svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{colors[band]}" stroke="#1f2937" stroke-width="1.4"/>')
        svg.append(f'<text x="{lx:.2f}" y="{ly-3:.2f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label_text[band][0]}</text>')
        svg.append(f'<text x="{lx:.2f}" y="{ly+16:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">{label_text[band][1]}</text>')

    svg.append("</svg>")
    (out_dir / "layerwise_fragility_selectivity.svg").write_text("\n".join(svg) + "\n")

    # Main paper version: make the selectivity claim explicit.
    # x = grounded collateral, y = hallucinated-minus-grounded fragility gap.
    y2_min, y2_max = -0.003, 0.022

    def sy2(y: float) -> float:
        return top + (y2_max - y) / (y2_max - y2_min) * plot_h

    svg2 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcff" stroke="#d7dee8" stroke-width="1"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Layer-wise fragility gap vs. grounded collateral</text>',
        f'<text x="{width/2}" y="56" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#64748b">x: grounded object drop; y: hallucinated-minus-grounded drop</text>',
    ]

    for x in [-0.005, 0.0, 0.005, 0.010, 0.015]:
        px = sx(x)
        svg2.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top+plot_h}" stroke="#e5e7eb" stroke-width="1"/>')
        svg2.append(f'<text x="{px:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">{x:.3f}</text>')
    for y in [0.0, 0.005, 0.010, 0.015, 0.020]:
        py = sy2(y)
        svg2.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left+plot_w}" y2="{py:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg2.append(f'<text x="{left-14}" y="{py+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#475569">{y:.3f}</text>')

    svg2.append(f'<line x1="{sx(0):.2f}" y1="{top}" x2="{sx(0):.2f}" y2="{top+plot_h}" stroke="#94a3b8" stroke-width="1.2"/>')
    svg2.append(f'<line x1="{left}" y1="{sy2(0):.2f}" x2="{left+plot_w}" y2="{sy2(0):.2f}" stroke="#94a3b8" stroke-width="1.2"/>')
    svg2.append(f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">Grounded object Δ log p collateral</text>')
    svg2.append(f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">H-G fragility gap Δ log p</text>')

    box_x, box_y = sx(0.0003), sy2(0.0202)
    svg2.append(f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="225" height="61" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>')
    svg2.append(f'<text x="{box_x+12:.2f}" y="{box_y+21:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">desired region:</text>')
    svg2.append(f'<text x="{box_x+12:.2f}" y="{box_y+39:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">large H-G fragility gap</text>')
    svg2.append(f'<text x="{box_x+12:.2f}" y="{box_y+57:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">low grounded collateral</text>')

    label_pos2 = {
        "L0-8": (-0.0032, 0.004),
        "L9-16": (0.0042, 0.0187),
        "L17-24": (0.0108, 0.0144),
        "L25-31": (0.0140, 0.0110),
    }
    label_text2 = {
        "L0-8": ["L0-8", "near zero"],
        "L9-16": ["L9-16", "largest gap, low cost"],
        "L17-24": ["L17-24", "smaller gap, higher cost"],
        "L25-31": ["L25-31", "late collateral"],
    }

    for row in rows:
        band = row["band"]
        x = row["grounded_delta"]
        y = row["gap"]
        px, py = sx(x), sy2(y)
        lx, ly = sx(label_pos2[band][0]), sy2(label_pos2[band][1])
        radius = 13 if band == "L9-16" else 10
        svg2.append(f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{lx-7:.2f}" y2="{ly:.2f}" stroke="#6b7280" stroke-width="1"/>')
        svg2.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{colors[band]}" stroke="#1f2937" stroke-width="1.4"/>')
        svg2.append(f'<text x="{lx:.2f}" y="{ly-3:.2f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label_text2[band][0]}</text>')
        svg2.append(f'<text x="{lx:.2f}" y="{ly+16:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#475569">{label_text2[band][1]}</text>')

    svg2.append("</svg>")
    (out_dir / "layerwise_fragility_gap_vs_collateral.svg").write_text("\n".join(svg2) + "\n")

    # Dense main figure: dumbbell plot. Each row shows grounded collateral,
    # hallucinated response, and the gap as the connecting segment.
    dw, dh = 980, 520
    dl, dr, dt, db = 150, 70, 88, 80
    pw, ph = dw - dl - dr, dh - dt - db
    dx_min, dx_max = -0.006, 0.032
    row_y = {
        "L25-31": dt + 52,
        "L17-24": dt + 132,
        "L9-16": dt + 212,
        "L0-8": dt + 292,
    }

    def dsx(x: float) -> float:
        return dl + (x - dx_min) / (dx_max - dx_min) * pw

    svg3 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{dw}" height="{dh}" viewBox="0 0 {dw} {dh}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{dw/2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Layer-wise text-side actuation is selective in L9-L16</text>',
        f'<text x="{dw/2}" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#64748b">Connected points show grounded collateral and hallucinated-object response under the same controlled perturbation</text>',
        f'<rect x="{dl}" y="{dt-24}" width="{pw}" height="{ph+40}" rx="12" fill="#fbfcff" stroke="#d7dee8"/>',
    ]
    for x in [-0.005, 0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
        px = dsx(x)
        svg3.append(f'<line x1="{px:.2f}" y1="{dt-24}" x2="{px:.2f}" y2="{dt+ph+16}" stroke="#e5e7eb" stroke-width="1"/>')
        svg3.append(f'<text x="{px:.2f}" y="{dt+ph+43}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#64748b">{x:.3f}</text>')
    svg3.append(f'<line x1="{dsx(0):.2f}" y1="{dt-24}" x2="{dsx(0):.2f}" y2="{dt+ph+16}" stroke="#94a3b8" stroke-width="1.4"/>')

    # Legend.
    legend_x = dl + pw - 300
    legend_y = 82
    svg3.extend(
        [
            f'<circle cx="{legend_x}" cy="{legend_y}" r="7" fill="#22c55e" stroke="#14532d" stroke-width="1"/>',
            f'<text x="{legend_x+14}" y="{legend_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">grounded drop</text>',
            f'<circle cx="{legend_x+125}" cy="{legend_y}" r="7" fill="#f97316" stroke="#7c2d12" stroke-width="1"/>',
            f'<text x="{legend_x+139}" y="{legend_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">hallucinated drop</text>',
            f'<line x1="{legend_x+248}" y1="{legend_y}" x2="{legend_x+282}" y2="{legend_y}" stroke="#7c3aed" stroke-width="4" stroke-linecap="round"/>',
            f'<text x="{legend_x+290}" y="{legend_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">H-G gap</text>',
        ]
    )

    row_map = {row["band"]: row for row in rows}
    for band in ["L25-31", "L17-24", "L9-16", "L0-8"]:
        row = row_map[band]
        y = row_y[band]
        gx, hx = dsx(row["grounded_delta"]), dsx(row["hall_delta"])
        gap = row["gap"]
        is_focus = band == "L9-16"
        if is_focus:
            svg3.append(f'<rect x="{dl-132}" y="{y-31}" width="{pw+162}" height="62" rx="12" fill="#f5f0ff" stroke="#c4b5fd" stroke-width="1.2"/>')
        svg3.append(f'<text x="{dl-20}" y="{y+5}" text-anchor="end" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#111827">{band}</text>')
        svg3.append(f'<line x1="{gx:.2f}" y1="{y}" x2="{hx:.2f}" y2="{y}" stroke="#7c3aed" stroke-width="{5 if is_focus else 3.5}" stroke-linecap="round"/>')
        svg3.append(f'<circle cx="{gx:.2f}" cy="{y}" r="{10 if is_focus else 8}" fill="#22c55e" stroke="#14532d" stroke-width="1.2"/>')
        svg3.append(f'<circle cx="{hx:.2f}" cy="{y}" r="{10 if is_focus else 8}" fill="#f97316" stroke="#7c2d12" stroke-width="1.2"/>')
        svg3.append(f'<text x="{gx:.2f}" y="{y-17}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#166534">{row["grounded_delta"]:.4f}</text>')
        svg3.append(f'<text x="{hx:.2f}" y="{y-17}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#9a3412">{row["hall_delta"]:.4f}</text>')
        svg3.append(f'<text x="{dl+pw+18}" y="{y-4}" font-family="Arial, sans-serif" font-size="12" fill="#475569">gap</text>')
        svg3.append(f'<text x="{dl+pw+18}" y="{y+15}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#7c3aed">{gap:.4f}</text>')
        if is_focus:
            svg3.append(f'<text x="{dl+295}" y="{y+35}" font-family="Arial, sans-serif" font-size="12" fill="#5b21b6">largest gap with near-zero grounded cost</text>')
        elif band == "L25-31":
            svg3.append(f'<text x="{gx+8:.2f}" y="{y+35}" font-family="Arial, sans-serif" font-size="12" fill="#64748b">larger grounded collateral</text>')

    svg3.append(f'<text x="{dl+pw/2}" y="{dh-24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">Object-token log-probability drop Δ log p</text>')
    svg3.append("</svg>")
    (out_dir / "layerwise_fragility_dumbbell.svg").write_text("\n".join(svg3) + "\n")

    # Vertical layout for paper columns: bands on x-axis, log-probability drop on y-axis.
    vw, vh = 720, 700
    vl, vr, vt, vb = 92, 50, 92, 84
    vpw, vph = vw - vl - vr, vh - vt - vb
    vy_min, vy_max = -0.006, 0.032
    bands = ["L0-8", "L9-16", "L17-24", "L25-31"]
    x_positions = {
        band: vl + (i + 0.5) * vpw / len(bands)
        for i, band in enumerate(bands)
    }

    def vsy(y: float) -> float:
        return vt + (vy_max - y) / (vy_max - vy_min) * vph

    svg4 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{vw}" height="{vh}" viewBox="0 0 {vw} {vh}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{vw/2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Layer-wise text-side actuation</text>',
        f'<text x="{vw/2}" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#64748b">L9-L16 shows the largest hallucinated-minus-grounded gap with low grounded collateral</text>',
        f'<rect x="{vl}" y="{vt}" width="{vpw}" height="{vph}" rx="12" fill="#fbfcff" stroke="#d7dee8"/>',
    ]

    # Highlight L9-L16 column.
    band_w = vpw / len(bands)
    focus_x = x_positions["L9-16"]
    svg4.append(
        f'<rect x="{focus_x-band_w/2+8:.2f}" y="{vt+8}" width="{band_w-16:.2f}" height="{vph-16}" rx="14" fill="#f5f0ff" stroke="#c4b5fd" stroke-width="1.5"/>'
    )

    for y in [-0.005, 0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
        py = vsy(y)
        svg4.append(f'<line x1="{vl}" y1="{py:.2f}" x2="{vl+vpw}" y2="{py:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg4.append(f'<text x="{vl-14}" y="{py+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{y:.3f}</text>')
    svg4.append(f'<line x1="{vl}" y1="{vsy(0):.2f}" x2="{vl+vpw}" y2="{vsy(0):.2f}" stroke="#94a3b8" stroke-width="1.4"/>')

    # Legend.
    legend_x, legend_y = vl + 28, 84
    svg4.extend(
        [
            f'<circle cx="{legend_x}" cy="{legend_y}" r="7" fill="#22c55e" stroke="#14532d" stroke-width="1"/>',
            f'<text x="{legend_x+14}" y="{legend_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">grounded drop</text>',
            f'<circle cx="{legend_x+142}" cy="{legend_y}" r="7" fill="#f97316" stroke="#7c2d12" stroke-width="1"/>',
            f'<text x="{legend_x+156}" y="{legend_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">hallucinated drop</text>',
            f'<line x1="{legend_x+313}" y1="{legend_y}" x2="{legend_x+348}" y2="{legend_y}" stroke="#7c3aed" stroke-width="4" stroke-linecap="round"/>',
            f'<text x="{legend_x+356}" y="{legend_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">H-G gap</text>',
        ]
    )

    row_map = {row["band"]: row for row in rows}
    for band in bands:
        row = row_map[band]
        x = x_positions[band]
        gy = vsy(row["grounded_delta"])
        hy = vsy(row["hall_delta"])
        is_focus = band == "L9-16"
        svg4.append(f'<text x="{x:.2f}" y="{vt+vph+34}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#111827">{band}</text>')
        svg4.append(
            f'<line x1="{x:.2f}" y1="{gy:.2f}" x2="{x:.2f}" y2="{hy:.2f}" stroke="#7c3aed" stroke-width="{7 if is_focus else 4}" stroke-linecap="round"/>'
        )
        svg4.append(f'<circle cx="{x:.2f}" cy="{gy:.2f}" r="{12 if is_focus else 9}" fill="#22c55e" stroke="#14532d" stroke-width="1.4"/>')
        svg4.append(f'<circle cx="{x:.2f}" cy="{hy:.2f}" r="{12 if is_focus else 9}" fill="#f97316" stroke="#7c2d12" stroke-width="1.4"/>')
        svg4.append(f'<text x="{x+16:.2f}" y="{gy+4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#166534">G {row["grounded_delta"]:.4f}</text>')
        svg4.append(f'<text x="{x+16:.2f}" y="{hy+4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#9a3412">H {row["hall_delta"]:.4f}</text>')
        svg4.append(f'<text x="{x:.2f}" y="{vt+vph+56}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#7c3aed">gap {row["gap"]:.4f}</text>')

    svg4.append(f'<text x="26" y="{vt+vph/2}" transform="rotate(-90 26 {vt+vph/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">Object-token log-probability drop Δ log p</text>')
    svg4.append("</svg>")
    (out_dir / "layerwise_fragility_dumbbell_vertical.svg").write_text("\n".join(svg4) + "\n")


if __name__ == "__main__":
    main()
