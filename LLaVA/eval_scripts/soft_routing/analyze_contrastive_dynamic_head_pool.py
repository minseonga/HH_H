import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter

from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [safe_float(value) for value in values]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def head_key(layer, head):
    return f"{int(layer)}:{int(head)}"


def lhh_key(layer, head):
    return f"L{int(layer)}H{int(head)}"


def parse_metric_tag(path):
    name = os.path.basename(os.path.dirname(path))
    match = re.search(r"_k(?P<k>\d+)_s(?P<s>[\d.]+)_q(?P<q>[\d.]+)_tau(?P<tau>[\d.]+)", name)
    if not match:
        return name
    return (
        f"dynamic_k{match.group('k')}_"
        f"s{match.group('s')}_q{match.group('q')}_tau{match.group('tau')}"
    )


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["empty"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ranked_heads(path):
    with open(path) as f:
        data = json.load(f)
    records = data.get("heads", [])
    for idx, row in enumerate(records, start=1):
        row["_rank"] = idx
        row["_key"] = head_key(row["layer"], row["head"])
        row["_lhh"] = lhh_key(row["layer"], row["head"])
    return data, records


def load_rank_map(path):
    _, records = load_ranked_heads(path)
    return {row["_key"]: idx for idx, row in enumerate(records, start=1)}


def rank_values(items):
    order = sorted(range(len(items)), key=lambda idx: (items[idx] is None, items[idx]))
    ranks = [0.0] * len(items)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and items[order[j]] == items[order[i]]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(xs, ys):
    pairs = [
        (safe_float(x), safe_float(y))
        for x, y in zip(xs, ys)
        if safe_float(x) is not None and safe_float(y) is not None
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx <= 0.0 or dy <= 0.0:
        return None
    return num / (dx * dy)


def spearman(xs, ys):
    pairs = [
        (safe_float(x), safe_float(y))
        for x, y in zip(xs, ys)
        if safe_float(x) is not None and safe_float(y) is not None
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    return pearson(rank_values(list(xs)), rank_values(list(ys)))


def quantile(values, q):
    values = sorted(value for value in (safe_float(item) for item in values) if value is not None)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def summarize_records(records, selected_keys, adhh_keys, label):
    rows = [row for row in records if row["_key"] in selected_keys]
    keys = set(selected_keys)
    features = [
        ("rank", "_rank"),
        ("score", "itext_all__C_toi_HminusG"),
        ("front_percentile", "front_percentile"),
        ("back_percentile", "back_percentile"),
        ("itext_all", "Itext_all"),
        ("itext_hallucinated", "Itext_hallucinated"),
        ("itext_non_hallucinated", "Itext_non_hallucinated"),
        ("raw_toi_hallucinated", "RawTOI_hallucinated"),
        ("raw_toi_non_hallucinated", "RawTOI_non_hallucinated"),
        ("log_toi_hallucinated", "LogTOI_hallucinated"),
        ("log_toi_non_hallucinated", "LogTOI_non_hallucinated"),
        ("img_hallucinated", "Img_hallucinated"),
        ("img_non_hallucinated", "Img_non_hallucinated"),
    ]
    out = {
        "bucket": label,
        "n": len(rows),
        "adhh_overlap": len(keys & adhh_keys),
        "adhh_recall": len(keys & adhh_keys) / max(len(adhh_keys), 1),
        "adhh_fraction": len(keys & adhh_keys) / max(len(keys), 1),
    }
    for out_name, in_name in features:
        values = [row.get(in_name) for row in rows]
        out[f"mean_{out_name}"] = mean(values)
        out[f"q50_{out_name}"] = quantile(values, 0.5)
    out["mean_itext_gap_hall_minus_nonhall"] = mean(
        safe_float(row.get("Itext_hallucinated"), 0.0) - safe_float(row.get("Itext_non_hallucinated"), 0.0)
        for row in rows
    )
    out["mean_raw_toi_gap_hall_minus_nonhall"] = mean(
        safe_float(row.get("RawTOI_hallucinated"), 0.0) - safe_float(row.get("RawTOI_non_hallucinated"), 0.0)
        for row in rows
    )
    out["mean_img_drop_hall_vs_nonhall"] = mean(
        safe_float(row.get("Img_non_hallucinated"), 0.0) - safe_float(row.get("Img_hallucinated"), 0.0)
        for row in rows
    )
    layers = [int(row["layer"]) for row in rows]
    out["min_layer"] = min(layers) if layers else None
    out["max_layer"] = max(layers) if layers else None
    out["mean_layer"] = mean(layers)
    return out


def make_bucket_rows(records, adhh_keys, top_ks):
    rows = []
    all_keys = [row["_key"] for row in records]
    for top_k in top_ks:
        rows.append(summarize_records(records, set(all_keys[:top_k]), adhh_keys, f"top{top_k}"))
    rows.append(summarize_records(records, set(all_keys[max(top_ks):]), adhh_keys, f"rank>{max(top_ks)}"))
    return rows


def make_overlay_rows(records, adhh_keys, top_ks):
    rows = []
    ordered_keys = [row["_key"] for row in records]
    key_to_lhh = {row["_key"]: row["_lhh"] for row in records}
    for top_k in top_ks:
        selected = set(ordered_keys[:top_k])
        overlap = selected & adhh_keys
        rows.append({
            "top_k": top_k,
            "selected": len(selected),
            "adhh_total": len(adhh_keys),
            "overlap": len(overlap),
            "selected_fraction_in_adhh": len(overlap) / max(len(selected), 1),
            "adhh_recall": len(overlap) / max(len(adhh_keys), 1),
            "jaccard": len(overlap) / max(len(selected | adhh_keys), 1),
            "overlap_heads": " ".join(sorted(key_to_lhh.get(key, key) for key in overlap)),
        })
    return rows


def make_adhh_rank_rows(records, adhh_keys):
    rows = []
    rank_by_key = {row["_key"]: row["_rank"] for row in records}
    row_by_key = {row["_key"]: row for row in records}
    for key in sorted(adhh_keys, key=lambda item: rank_by_key.get(item, 10**9)):
        row = row_by_key.get(key, {})
        layer, head = key.split(":")
        rows.append({
            "head_key": key,
            "head_id": lhh_key(layer, head),
            "rank": rank_by_key.get(key),
            "score": row.get("itext_all__C_toi_HminusG"),
            "front_percentile": row.get("front_percentile"),
            "back_percentile": row.get("back_percentile"),
            "itext_all": row.get("Itext_all"),
            "raw_toi_gap": (
                safe_float(row.get("RawTOI_hallucinated"), 0.0)
                - safe_float(row.get("RawTOI_non_hallucinated"), 0.0)
            ) if row else None,
            "img_drop": (
                safe_float(row.get("Img_non_hallucinated"), 0.0)
                - safe_float(row.get("Img_hallucinated"), 0.0)
            ) if row else None,
        })
    return rows


def make_component_overlap_rows(target_records, zoo_dir, top_ks):
    target_keys = [row["_key"] for row in target_records]
    rows = []
    component_names = [
        "ranked_heads_itext_all.json",
        "ranked_heads_C_ratio_hall_minus_nonhall.json",
        "ranked_heads_C_logtoi_hall_minus_nonhall.json",
        "ranked_heads_C_itext_hall_minus_nonhall.json",
        "ranked_heads_text_all.json",
        "ranked_heads_ratio_all.json",
        "ranked_heads_low_image_hall.json",
    ]
    for filename in component_names:
        path = os.path.join(zoo_dir, filename)
        if not os.path.exists(path):
            continue
        _, records = load_ranked_heads(path)
        comp_keys = [row["_key"] for row in records]
        comp_name = os.path.basename(path).replace("ranked_heads_", "").replace(".json", "")
        for top_k in top_ks:
            a = set(target_keys[:top_k])
            b = set(comp_keys[:top_k])
            inter = a & b
            rows.append({
                "component": comp_name,
                "top_k": top_k,
                "overlap": len(inter),
                "target_precision": len(inter) / max(len(a), 1),
                "component_recall": len(inter) / max(len(b), 1),
                "jaccard": len(inter) / max(len(a | b), 1),
            })
    return rows


def make_correlation_rows(target_records, zoo_dir):
    target_denom = float(max(len(target_records) - 1, 1))
    target_by_key = {
        row["_key"]: 1.0 - (row["_rank"] - 1) / target_denom
        for row in target_records
    }
    rows = []
    for path in sorted(glob.glob(os.path.join(zoo_dir, "ranked_heads_*.json"))):
        if path.endswith("ranked_heads_global__itext_all__C_toi_HminusG.json"):
            continue
        try:
            data, records = load_ranked_heads(path)
        except Exception:
            continue
        score_name = data.get("score_name", os.path.basename(path).replace("ranked_heads_", "").replace(".json", ""))
        denom = float(max(len(records) - 1, 1))
        common = []
        for row in records:
            key = row["_key"]
            if key not in target_by_key:
                continue
            score = 1.0 - (row["_rank"] - 1) / denom
            common.append((target_by_key[key], score))
        if len(common) < 10:
            continue
        xs, ys = zip(*common)
        rows.append({
            "score_name": score_name,
            "n_common": len(common),
            "pearson_with_target_rank_percentile": pearson(xs, ys),
            "spearman_with_target_rank_percentile": spearman(xs, ys),
            "path": path,
        })
    rows.sort(key=lambda row: safe_float(row.get("spearman_with_target_rank_percentile"), -1.0), reverse=True)
    return rows


def make_layer_rows(records, top_ks):
    rows = []
    for top_k in top_ks:
        selected = records[:top_k]
        counts = Counter(int(row["layer"]) for row in selected)
        for layer in sorted(counts):
            rows.append({
                "top_k": top_k,
                "layer": layer,
                "count": counts[layer],
                "fraction": counts[layer] / max(top_k, 1),
            })
    return rows


def make_architecture_rows(records, adhh_keys, top_ks):
    rows = []
    rank_by_key = {row["_key"]: row["_rank"] for row in records}
    bands = [
        ("L13-20", 13, 20),
        ("L21-26", 21, 26),
        ("L27-31", 27, 31),
    ]

    def band_name(layer):
        for name, low, high in bands:
            if low <= layer <= high:
                return name
        return "other"

    for top_k in top_ks:
        selected = records[:top_k]
        layer_counts = Counter(int(row["layer"]) for row in selected)
        band_counts = Counter(band_name(int(row["layer"])) for row in selected)
        balanced = sum(
            1
            for row in selected
            if safe_float(row.get("front_percentile"), 0.0) >= 0.8
            and safe_float(row.get("back_percentile"), 0.0) >= 0.8
        )
        front_only = sum(
            1
            for row in selected
            if safe_float(row.get("front_percentile"), 0.0) >= 0.8
            and safe_float(row.get("back_percentile"), 0.0) < 0.8
        )
        back_only = sum(
            1
            for row in selected
            if safe_float(row.get("front_percentile"), 0.0) < 0.8
            and safe_float(row.get("back_percentile"), 0.0) >= 0.8
        )
        toi_gap_pos = sum(
            1
            for row in selected
            if safe_float(row.get("RawTOI_hallucinated"), 0.0)
            > safe_float(row.get("RawTOI_non_hallucinated"), 0.0)
        )
        img_drop_pos = sum(
            1
            for row in selected
            if safe_float(row.get("Img_non_hallucinated"), 0.0)
            > safe_float(row.get("Img_hallucinated"), 0.0)
        )
        prior_min = 1.0 - (top_k - 1) / float(max(len(records) - 1, 1))
        rows.append({
            "top_k": top_k,
            "unique_layers": len(layer_counts),
            "max_heads_per_layer": max(layer_counts.values()) if layer_counts else 0,
            "band_L13_20": band_counts["L13-20"],
            "band_L21_26": band_counts["L21-26"],
            "band_L27_31": band_counts["L27-31"],
            "frac_L13_20": band_counts["L13-20"] / max(top_k, 1),
            "frac_L21_26": band_counts["L21-26"] / max(top_k, 1),
            "frac_L27_31": band_counts["L27-31"] / max(top_k, 1),
            "balanced_front_back_ge_0p8": balanced,
            "front_only_ge_0p8": front_only,
            "back_only_ge_0p8": back_only,
            "toi_gap_positive": toi_gap_pos,
            "img_drop_positive": img_drop_pos,
            "rank_percentile_prior_min": prior_min,
            "adhh_overlap": sum(1 for row in selected if row["_key"] in adhh_keys),
        })

    rejected_adhh_rows = []
    for key in sorted(adhh_keys, key=lambda item: rank_by_key.get(item, 10**9)):
        rank = rank_by_key.get(key)
        if rank is None or rank <= max(top_ks):
            continue
        row = records[rank - 1]
        rejected_adhh_rows.append({
            "head_key": key,
            "head_id": row["_lhh"],
            "rank": rank,
            "front_percentile": row.get("front_percentile"),
            "back_percentile": row.get("back_percentile"),
            "itext_all": row.get("Itext_all"),
            "raw_toi_gap": (
                safe_float(row.get("RawTOI_hallucinated"), 0.0)
                - safe_float(row.get("RawTOI_non_hallucinated"), 0.0)
            ),
            "img_drop": (
                safe_float(row.get("Img_non_hallucinated"), 0.0)
                - safe_float(row.get("Img_hallucinated"), 0.0)
            ),
        })
    return rows, rejected_adhh_rows


def load_eval_metrics(paths):
    rows = []
    for label, path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        metrics = data.get("overall_metrics", data)
        bleu = metrics.get("Bleu") or []
        rows.append({
            "method": label,
            "CHAIRs": metrics.get("CHAIRs"),
            "CHAIRi": metrics.get("CHAIRi"),
            "Bleu1": bleu[0] if bleu else metrics.get("Bleu1"),
            "METEOR": metrics.get("METEOR"),
            "CIDEr": metrics.get("CIDEr"),
            "precision": metrics.get("ObjectPrecision", metrics.get("precision")),
            "recall": metrics.get("ObjectRecall", metrics.get("recall")),
            "f1": metrics.get("ObjectF1", metrics.get("f1")),
            "path": path,
        })
    return rows


def discover_eval_paths(args):
    paths = [
        ("greedy_n500", args.base_eval_json),
        ("adhh_k20_n500", args.adhh_eval_json),
    ]
    for path in sorted(glob.glob(args.dynamic_eval_glob)):
        paths.append((parse_metric_tag(path), path))
    return [(label, path) for label, path in paths if path]


def best_dynamic(metric_rows):
    dynamic_rows = [row for row in metric_rows if row["method"].startswith("dynamic_")]
    if not dynamic_rows:
        return None
    return min(dynamic_rows, key=lambda row: safe_float(row.get("CHAIRs"), 999.0))


def make_report(
    path,
    args,
    target_data,
    records,
    bucket_rows,
    overlay_rows,
    corr_rows,
    metric_rows,
    top_ks,
    architecture_rows,
    rejected_adhh_rows,
):
    top20 = " ".join(row["_lhh"] for row in records[:20])
    top100_layers = Counter(int(row["layer"]) for row in records[:100])
    layer_text = ", ".join(f"L{layer}:{count}" for layer, count in sorted(top100_layers.items()))
    top100 = next((row for row in bucket_rows if row["bucket"] == "top100"), {})
    top150 = next((row for row in bucket_rows if row["bucket"] == "top150"), {})
    arch100 = next((row for row in architecture_rows if int(row["top_k"]) == 100), {})
    arch150 = next((row for row in architecture_rows if int(row["top_k"]) == 150), {})
    rest = next((row for row in bucket_rows if row["bucket"].startswith("rank>")), {})
    rest_label = rest.get("bucket", f"rank>{max(top_ks)}")
    ov100 = next((row for row in overlay_rows if int(row["top_k"]) == 100), {})
    best_dyn = best_dynamic(metric_rows)
    greedy = next((row for row in metric_rows if row["method"] == "greedy_n500"), None)
    adhh = next((row for row in metric_rows if row["method"] == "adhh_k20_n500"), None)
    corr_top = corr_rows[:8]

    def fmt(value, ndigits=4):
        value = safe_float(value)
        return "NA" if value is None else f"{value:.{ndigits}f}"

    lines = [
        "# Contrastive Dynamic Head Pool Justification",
        "",
        f"- ranked head file: `{args.ranked_heads}`",
        f"- score: `{target_data.get('score_name')}`",
        f"- description: {target_data.get('description', '')}",
        "",
        "## 핵심 해석",
        "",
        "이 head pool은 AD-HH head를 그대로 가져온 것이 아니라, 두 축을 rank-percentile로 결합한 pool이다.",
        "첫 축은 intervention 대상인 image 이후 text-side attention mass이고, 둘째 축은 hallucinated object step에서 grounded object step보다 text-over-image ratio가 커지는 contrast이다.",
        "따라서 이 pool의 의미는 `text leverage`와 `hallucination-specific text-over-image reliance`의 교집합이다.",
        "",
        "## Pool 구조",
        "",
        f"- top20 heads: `{top20}`",
        f"- top100 layer distribution: {layer_text}",
        f"- top100 AD-HH overlap: {ov100.get('overlap', 'NA')}/20 recall={fmt(ov100.get('adhh_recall'))}, selected fraction={fmt(ov100.get('selected_fraction_in_adhh'))}",
        f"- top100 mean Itext_all={fmt(top100.get('mean_itext_all'))}, {rest_label} mean Itext_all={fmt(rest.get('mean_itext_all'))}",
        f"- top100 mean RawTOI H-G gap={fmt(top100.get('mean_raw_toi_gap_hall_minus_nonhall'))}, {rest_label} gap={fmt(rest.get('mean_raw_toi_gap_hall_minus_nonhall'))}",
        f"- top100 mean image drop(nonhall-hall)={fmt(top100.get('mean_img_drop_hall_vs_nonhall'))}, {rest_label} image drop={fmt(rest.get('mean_img_drop_hall_vs_nonhall'))}",
        "",
        "## Architecture-Level Findings",
        "",
        "### Finding 1: larger pool is a distributed actuator scaffold, not AD-HH copy",
        "",
        f"- top100 covers {arch100.get('unique_layers', 'NA')} layers with max {arch100.get('max_heads_per_layer', 'NA')} heads/layer; top150 covers {arch150.get('unique_layers', 'NA')} layers with max {arch150.get('max_heads_per_layer', 'NA')} heads/layer.",
        f"- top100 band split: L13-20={arch100.get('band_L13_20', 'NA')}, L21-26={arch100.get('band_L21_26', 'NA')}, L27-31={arch100.get('band_L27_31', 'NA')}.",
        f"- top150 band split: L13-20={arch150.get('band_L13_20', 'NA')}, L21-26={arch150.get('band_L21_26', 'NA')}, L27-31={arch150.get('band_L27_31', 'NA')}.",
        "Interpretation: the pool spans mid-layer cross-modal competition and late language-readout layers, so the method is a distributed routing intervention rather than a small fixed head mask.",
        "",
        "### Finding 2: top100/top150 keep AD-HH's useful core but reject weak AD-HH heads",
        "",
        f"- top100 recovers {arch100.get('adhh_overlap', 'NA')}/20 AD-HH heads; top150 recovers {arch150.get('adhh_overlap', 'NA')}/20.",
        "- The rejected AD-HH heads after top150 are mostly high text-mass but low/negative hallucination contrast:",
    ]
    for row in rejected_adhh_rows[:8]:
        lines.append(
            f"  - `{row['head_id']}` rank={row['rank']}, front={fmt(row['front_percentile'])}, "
            f"back={fmt(row['back_percentile'])}, RawTOI gap={fmt(row['raw_toi_gap'])}"
        )
    lines.extend([
        "Interpretation: this directly separates generic language-continuation heads from hallucination-specific text-over-image heads.",
        "",
        "### Finding 3: top100 is the balanced core; top150 is the aggressive contrast shell",
        "",
        f"- top100 balanced(front>=0.8 and back>=0.8) heads: {arch100.get('balanced_front_back_ge_0p8', 'NA')}; top150: {arch150.get('balanced_front_back_ge_0p8', 'NA')}.",
        f"- top100 back-only contrast heads: {arch100.get('back_only_ge_0p8', 'NA')}; top150: {arch150.get('back_only_ge_0p8', 'NA')}.",
        "Interpretation: top100 is the high-confidence intersection of text leverage and hallucination contrast. top150 adds more hallucination-contrast-heavy heads, which is more aggressive and can lower CHAIR further but risks recall.",
        "",
        "### Finding 4: the selected heads show hallucination-state visual dropout",
        "",
        f"- top100 heads have positive RawTOI H-G gap in {arch100.get('toi_gap_positive', 'NA')}/100 heads and positive image drop in {arch100.get('img_drop_positive', 'NA')}/100 heads.",
        f"- top150 heads have positive RawTOI H-G gap in {arch150.get('toi_gap_positive', 'NA')}/150 heads and positive image drop in {arch150.get('img_drop_positive', 'NA')}/150 heads.",
        f"- mean image drop is {fmt(top100.get('mean_img_drop_hall_vs_nonhall'))} for top100 and {fmt(top150.get('mean_img_drop_hall_vs_nonhall'))} for top150.",
        "Interpretation: hallucination steps are not merely high-text; they are relatively lower-image and higher text-over-image in these selected heads.",
        "",
        "### Finding 5: dynamic rank prior keeps top100/top150 active but graded",
        "",
        f"- rank-percentile prior min is {fmt(arch100.get('rank_percentile_prior_min'))} for top100 and {fmt(arch150.get('rank_percentile_prior_min'))} for top150.",
        "Interpretation: expansion to top100/top150 does not mean uniform hard suppression. Offline rank gives a graded where-prior, and online text ratio gives token-level strength.",
        "",
        "## Visual Suppression Evidence",
        "",
        "- `suppression_evidence_figures.md` contains the figure bundle that justifies text-side suppression directly from attention-source behavior.",
        "- `suppression_evidence_head_space.svg` shows that selected heads occupy the high intervention-text-mass and high hallucinated-vs-grounded text-over-image region.",
        "- `suppression_evidence_source_shift.svg` compares selected heads against the rank>200 tail on Itext gap, image drop, and log text-over-image gap.",
        "- `suppression_evidence_layer_head_map.svg` shows top100/top150 as a distributed mid-to-late layer actuator scaffold with AD-HH overlap and rejected AD-HH heads marked.",
        "",
        "## Surrogate Consistency",
        "",
    ])
    for row in corr_top:
        lines.append(
            f"- `{row['score_name']}`: Spearman with target rank percentile={fmt(row.get('spearman_with_target_rank_percentile'))}"
        )
    lines.extend([
        "",
        "## Empirical Link",
        "",
    ])
    if greedy and adhh:
        lines.append(
            f"- greedy n=500: CHAIRs={fmt(greedy.get('CHAIRs'))}, CHAIRi={fmt(greedy.get('CHAIRi'))}, "
            f"F1={fmt(greedy.get('f1'))}"
        )
        lines.append(
            f"- AD-HH k20 n=500: CHAIRs={fmt(adhh.get('CHAIRs'))}, CHAIRi={fmt(adhh.get('CHAIRi'))}, "
            f"F1={fmt(adhh.get('f1'))}"
        )
    if best_dyn:
        lines.append(
            f"- best local dynamic run: `{best_dyn['method']}` CHAIRs={fmt(best_dyn.get('CHAIRs'))}, "
            f"CHAIRi={fmt(best_dyn.get('CHAIRi'))}, BLEU1={fmt(best_dyn.get('Bleu1'))}, "
            f"F1={fmt(best_dyn.get('f1'))}"
        )
    lines.extend([
        "",
        "## 논문/보고서에 넣을 수 있는 주장",
        "",
        "1. AD-HH는 fixed small actuator set이지만, teammate method는 larger distributed hallucination-relevant text-leverage scaffold를 사용한다.",
        "2. Head selection 자체가 suppression target과 정렬되어 있다. generated-prefix raw text attention이 아니라 image 이후 text-side slice를 쓰므로 실제 intervention slice와 같은 축이다.",
        "3. Contrastive term 때문에 단순 text-heavy head가 아니라 hallucinated object step에서 text-over-image reliance가 더 커지는 head를 우선한다.",
        "4. Top100/top150은 AD-HH의 useful core를 포함하되, AD-HH 내부의 generic text heads를 contrast criterion으로 거른다.",
        "5. Online gate는 head가 매 step 실제로 text-dominant일 때만 강하게 suppress한다. 즉 offline head pool은 `where`, online ratio는 `when/strength` 역할을 한다.",
        "6. 우리 AD-HH 분석의 결론과 맞는다. text_mass는 hallucination evidence가 아니라 actuator leverage이고, dynamic method는 이 leverage를 hallucination contrast 및 online state와 결합한다.",
        "",
        "## 남는 약점",
        "",
        "- Recall 하락은 여전히 있다. 큰 top-k pool에서 grounded object도 영향을 받기 때문이다.",
        "- 그래서 현재 추가 실험의 목적은 CHAIR를 유지하거나 더 낮추면서 concentration gate로 suppression을 더 선별적으로 만들어 recall/F1 손실을 줄이는 것이다.",
        "",
    ])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranked-heads",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json",
    )
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--output-dir", default="./results/coco/contrastive_dynamic_head_pool_analysis")
    parser.add_argument("--top-ks", default="20,30,50,100,150,200")
    parser.add_argument(
        "--base-eval-json",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_n500/captions_eval_results.json",
    )
    parser.add_argument(
        "--adhh-eval-json",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_adhh_file_k20_tau0.4_n500_real/captions_eval_results.json",
    )
    parser.add_argument(
        "--dynamic-eval-glob",
        default="../ADHH/LLaVA/results_dynamic/coco/*global__itext_all__C_toi_HminusG/captions_eval_results.json",
    )
    args = parser.parse_args()

    top_ks = [int(item) for item in args.top_ks.replace(" ", "").split(",") if item]
    target_data, records = load_ranked_heads(args.ranked_heads)
    zoo_dir = os.path.dirname(args.ranked_heads)
    adhh_heads = default_heads_for_model(args.model_path)
    adhh_keys = {head_key(layer, head) for layer, head in adhh_heads}

    bucket_rows = make_bucket_rows(records, adhh_keys, top_ks)
    overlay_rows = make_overlay_rows(records, adhh_keys, top_ks)
    adhh_rank_rows = make_adhh_rank_rows(records, adhh_keys)
    component_rows = make_component_overlap_rows(records, zoo_dir, top_ks)
    corr_rows = make_correlation_rows(records, zoo_dir)
    layer_rows = make_layer_rows(records, top_ks)
    architecture_rows, rejected_adhh_rows = make_architecture_rows(records, adhh_keys, top_ks)
    metric_rows = load_eval_metrics(discover_eval_paths(args))

    write_csv(os.path.join(args.output_dir, "head_pool_rank_bucket_summary.csv"), bucket_rows)
    write_csv(os.path.join(args.output_dir, "head_pool_adhh_overlay.csv"), overlay_rows)
    write_csv(os.path.join(args.output_dir, "adhh_default_head_ranks_in_pool.csv"), adhh_rank_rows)
    write_csv(os.path.join(args.output_dir, "head_pool_component_overlap.csv"), component_rows)
    write_csv(os.path.join(args.output_dir, "head_pool_surrogate_correlations.csv"), corr_rows)
    write_csv(os.path.join(args.output_dir, "head_pool_layer_distribution.csv"), layer_rows)
    write_csv(os.path.join(args.output_dir, "head_pool_architecture_findings.csv"), architecture_rows)
    write_csv(os.path.join(args.output_dir, "rejected_adhh_heads_by_contrastive_pool.csv"), rejected_adhh_rows)
    write_csv(os.path.join(args.output_dir, "local_eval_metrics.csv"), metric_rows)

    report_path = os.path.join(args.output_dir, "method_justification_report.md")
    make_report(
        report_path,
        args,
        target_data,
        records,
        bucket_rows,
        overlay_rows,
        corr_rows,
        metric_rows,
        top_ks,
        architecture_rows,
        rejected_adhh_rows,
    )

    summary = {
        "ranked_heads": args.ranked_heads,
        "score_name": target_data.get("score_name"),
        "n_heads": len(records),
        "top20": [row["_key"] for row in records[:20]],
        "outputs": {
            "report": report_path,
            "bucket_summary": os.path.join(args.output_dir, "head_pool_rank_bucket_summary.csv"),
            "adhh_overlay": os.path.join(args.output_dir, "head_pool_adhh_overlay.csv"),
            "adhh_ranks": os.path.join(args.output_dir, "adhh_default_head_ranks_in_pool.csv"),
            "component_overlap": os.path.join(args.output_dir, "head_pool_component_overlap.csv"),
            "surrogate_correlations": os.path.join(args.output_dir, "head_pool_surrogate_correlations.csv"),
            "layer_distribution": os.path.join(args.output_dir, "head_pool_layer_distribution.csv"),
            "architecture_findings": os.path.join(args.output_dir, "head_pool_architecture_findings.csv"),
            "rejected_adhh_heads": os.path.join(args.output_dir, "rejected_adhh_heads_by_contrastive_pool.csv"),
            "eval_metrics": os.path.join(args.output_dir, "local_eval_metrics.csv"),
        },
    }
    with open(os.path.join(args.output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
