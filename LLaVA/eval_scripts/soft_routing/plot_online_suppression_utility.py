import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURES_TO_PLOT = [
    "text_value_norm",
    "sample_norm_text_value_norm",
    "sample_norm_value_x_low_visual",
    "text_mass",
    "text_ratio",
    "visual_value_ratio",
]


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def load_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_label(label):
    if str(label).startswith("hallucinated"):
        return "hallucinated"
    if label in {"lost_grounded", "kept_grounded"}:
        return "grounded"
    return str(label)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_auc_bars(auc_rows, output_dir, group="all", top_n=12):
    rows = [row for row in auc_rows if row.get("group") == group]
    rows = sorted(rows, key=lambda row: safe_float(row.get("auroc_abs")), reverse=True)[:top_n]
    if not rows:
        return None

    labels = [row["feature"] for row in rows]
    aucs = [safe_float(row["auroc_abs"]) for row in rows]
    directions = [row.get("direction", "") for row in rows]
    colors = ["#d95f02" if direction.startswith("high") else "#1b9e77" for direction in directions]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(labels))
    ax.barh(y, aucs, color=colors, alpha=0.9)
    ax.axvline(0.5, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.45, max(0.85, max(aucs) + 0.03))
    ax.set_xlabel("AUROC, direction corrected")
    ax.set_title(f"Online Suppression Utility Feature Ranking ({group})")
    for idx, value in enumerate(aucs):
        ax.text(value + 0.006, idx, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, f"auc_bar_{group}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def rows_for_group(rows, group):
    if group == "all":
        return rows
    if group == "grounded":
        return [row for row in rows if normalize_label(row.get("label_family")) == "grounded"]
    return [row for row in rows if normalize_label(row.get("label_family")) == group]


def plot_feature_distributions(rows, output_dir, feature="text_value_norm", group="all"):
    group_rows = rows_for_group(rows, group)
    pos = [safe_float(row.get(feature)) for row in group_rows if str(row.get("should_suppress")).lower() == "true"]
    neg = [safe_float(row.get(feature)) for row in group_rows if str(row.get("should_suppress")).lower() != "true"]
    if len(pos) < 2 or len(neg) < 2:
        return None

    high = np.percentile(np.array(pos + neg), 99)
    low = np.percentile(np.array(pos + neg), 1)
    bins = np.linspace(low, high, 35)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(neg, bins=bins, density=True, alpha=0.55, color="#7570b3", label="do not suppress")
    ax.hist(pos, bins=bins, density=True, alpha=0.55, color="#d95f02", label="should suppress")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.set_title(f"{feature}: should-suppress separation ({group})")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(output_dir, f"distribution_{feature}_{group}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def bin_curve(rows, feature, group, n_bins):
    group_rows = rows_for_group(rows, group)
    values = np.array([safe_float(row.get(feature)) for row in group_rows], dtype=float)
    if len(values) < n_bins or np.max(values) <= np.min(values):
        return []
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    out = []
    for idx in range(n_bins):
        low, high = edges[idx], edges[idx + 1]
        bin_rows = [
            row for row in group_rows
            if safe_float(row.get(feature)) > low and safe_float(row.get(feature)) <= high
        ]
        if not bin_rows:
            continue
        utilities = [safe_float(row.get("suppression_utility")) for row in bin_rows]
        positives = [str(row.get("should_suppress")).lower() == "true" for row in bin_rows]
        out.append({
            "group": group,
            "feature": feature,
            "bin": idx + 1,
            "n": len(bin_rows),
            "feature_mean": float(np.mean([safe_float(row.get(feature)) for row in bin_rows])),
            "feature_low": float(low),
            "feature_high": float(high),
            "positive_rate": float(np.mean(positives)),
            "mean_utility": float(np.mean(utilities)),
        })
    return out


def plot_bin_curves(rows, output_dir, feature="text_value_norm", n_bins=10):
    groups = ["hallucinated", "grounded", "all"]
    curves = {group: bin_curve(rows, feature, group, n_bins) for group in groups}
    if not any(curves.values()):
        return None, []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=False)
    colors = {"hallucinated": "#d95f02", "grounded": "#1b9e77", "all": "#7570b3"}
    for group, points in curves.items():
        if not points:
            continue
        x = [point["feature_mean"] for point in points]
        axes[0].plot(x, [point["positive_rate"] for point in points], marker="o", color=colors[group], label=group)
        axes[1].plot(x, [point["mean_utility"] for point in points], marker="o", color=colors[group], label=group)

    axes[0].set_title("Positive suppression rate")
    axes[0].set_ylabel("P(utility > threshold)")
    axes[0].set_xlabel(feature)
    axes[1].set_title("Mean suppression utility")
    axes[1].set_ylabel("mean utility")
    axes[1].set_xlabel(feature)
    for ax in axes:
        ax.axhline(0.0, color="#555555", linewidth=1, linestyle="--", alpha=0.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(output_dir, f"bin_curve_{feature}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)

    rows_out = []
    for points in curves.values():
        rows_out.extend(points)
    return path, rows_out


def plot_text_mass_vs_value(rows, output_dir):
    if not rows:
        return None
    groups = {
        "hallucinated": [row for row in rows if normalize_label(row.get("label_family")) == "hallucinated"],
        "grounded": [row for row in rows if normalize_label(row.get("label_family")) == "grounded"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True, sharey=True)
    for ax, (group, group_rows) in zip(axes, groups.items()):
        x = np.array([safe_float(row.get("text_mass")) for row in group_rows])
        y = np.array([safe_float(row.get("text_value_norm")) for row in group_rows])
        utility = np.array([safe_float(row.get("suppression_utility")) for row in group_rows])
        if len(x) == 0:
            continue
        lo, hi = np.percentile(utility, [5, 95])
        scale = max(abs(lo), abs(hi), 1e-6)
        sc = ax.scatter(
            x,
            y,
            c=np.clip(utility, -scale, scale),
            cmap="coolwarm",
            s=14,
            alpha=0.72,
            linewidths=0,
            vmin=-scale,
            vmax=scale,
        )
        ax.axvline(0.4, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(group)
        ax.set_xlabel("text_mass")
        ax.set_ylabel("text_value_norm")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("suppression utility")
    fig.suptitle("Text Mass Alone Misses Value Contribution")
    fig.tight_layout()
    path = os.path.join(output_dir, "scatter_text_mass_vs_text_value_norm.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def feature_summary(rows, features):
    out = []
    for group in ["all", "hallucinated", "grounded"]:
        group_rows = rows_for_group(rows, group)
        for feature in features:
            pos = [safe_float(row.get(feature)) for row in group_rows if str(row.get("should_suppress")).lower() == "true"]
            neg = [safe_float(row.get(feature)) for row in group_rows if str(row.get("should_suppress")).lower() != "true"]
            if not pos or not neg:
                continue
            out.append({
                "group": group,
                "feature": feature,
                "n_positive": len(pos),
                "n_negative": len(neg),
                "mean_positive": float(np.mean(pos)),
                "mean_negative": float(np.mean(neg)),
                "median_positive": float(np.median(pos)),
                "median_negative": float(np.median(neg)),
                "mean_gap": float(np.mean(pos) - np.mean(neg)),
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--feature", default="text_value_norm")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    utility_dir = args.utility_dir
    output_dir = args.output_dir or os.path.join(utility_dir, "figures")
    ensure_dir(output_dir)

    rows_path = os.path.join(utility_dir, "suppression_utility_rows.csv")
    auc_path = os.path.join(utility_dir, "feature_suppression_utility_auc.csv")
    if not os.path.exists(rows_path):
        raise FileNotFoundError(rows_path)
    if not os.path.exists(auc_path):
        raise FileNotFoundError(auc_path)

    rows = load_csv(rows_path)
    auc_rows = load_csv(auc_path)

    paths = []
    for group in ["all", "hallucinated", "grounded"]:
        path = plot_auc_bars(auc_rows, output_dir, group=group)
        if path:
            paths.append(path)
    for group in ["all", "hallucinated", "grounded"]:
        path = plot_feature_distributions(rows, output_dir, feature=args.feature, group=group)
        if path:
            paths.append(path)
    path, curve_rows = plot_bin_curves(rows, output_dir, feature=args.feature, n_bins=args.bins)
    if path:
        paths.append(path)
    path = plot_text_mass_vs_value(rows, output_dir)
    if path:
        paths.append(path)

    write_csv(os.path.join(output_dir, "binned_utility_curve.csv"), curve_rows)
    write_csv(os.path.join(output_dir, "feature_distribution_summary.csv"), feature_summary(rows, FEATURES_TO_PLOT))

    summary = {
        "utility_dir": utility_dir,
        "output_dir": output_dir,
        "num_rows": len(rows),
        "figures": paths,
    }
    with open(os.path.join(output_dir, "plot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
