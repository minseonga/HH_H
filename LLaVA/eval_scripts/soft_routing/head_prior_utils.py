import json


DEFAULT_LLAVA15_HEADS = [[16, 29], [26, 9], [13, 31], [15, 10], [20, 12], [30, 9], [19, 18], [17, 0], [18, 9], [26, 28],
                         [19, 27], [18, 26], [15, 25], [14, 16], [31, 26], [15, 24], [31, 3], [22, 20], [27, 29], [17, 28]]

DEFAULT_LLAVA15_13B_HEADS = [[0, 8], [29, 27], [23, 18], [20, 11], [36, 26], [19, 37], [22, 16], [22, 34], [21, 31], [20, 34],
                             [37, 11], [17, 25], [35, 10], [17, 5], [15, 26], [0, 22], [19, 5], [19, 0], [14, 1], [23, 20],
                             [21, 6], [30, 24], [26, 27], [21, 32], [15, 28], [15, 31], [19, 30], [20, 8], [19, 14], [14, 9],
                             [39, 26], [25, 1], [18, 32], [17, 27], [39, 32]]

DEFAULT_LLAVA16_34B_HEADS = [[45, 34], [43, 4], [43, 48], [44, 29], [35, 47], [40, 27], [54, 34], [37, 48], [43, 2], [41, 34]]


def default_heads_for_model(model_path):
    if model_path == "liuhaotian/llava-v1.5-13b":
        return DEFAULT_LLAVA15_13B_HEADS
    if model_path == "liuhaotian/llava-v1.6-34b":
        return DEFAULT_LLAVA16_34B_HEADS
    return DEFAULT_LLAVA15_HEADS


def head_key(layer, head):
    return f"{int(layer)}:{int(head)}"


def parse_head_key(key):
    layer, head = key.split(":")
    return int(layer), int(head)


def rank_priors(heads):
    if len(heads) == 1:
        return {head_key(*heads[0]): 1.0}
    priors = {}
    for idx, head in enumerate(heads):
        priors[head_key(*head)] = max(0.0, 1.0 - idx / float(len(heads) - 1))
    return priors


def score_priors(score_items):
    if not score_items:
        return {}
    values = [float(item["score"]) for item in score_items]
    sorted_values = sorted(values)
    q50 = sorted_values[int(round((len(sorted_values) - 1) * 0.50))]
    q95 = sorted_values[int(round((len(sorted_values) - 1) * 0.95))]
    denom = max(q95 - q50, 1e-8)
    priors = {}
    for item in score_items:
        prior = (float(item["score"]) - q50) / denom
        priors[head_key(item["layer"], item["head"])] = min(1.0, max(0.0, prior))
    return priors


def uniform_priors(heads):
    return {head_key(*head): 1.0 for head in heads}


def _score_items_from_field(raw_scores):
    score_items = []
    if isinstance(raw_scores, dict):
        for key, score in raw_scores.items():
            layer, head = parse_head_key(key)
            score_items.append({"layer": layer, "head": head, "score": score})
    else:
        for item in raw_scores:
            if isinstance(item, dict):
                score_items.append(item)
            else:
                score_items.append({"layer": item[0], "head": item[1], "score": item[2]})
    return score_items


def load_head_priors(path=None, top_k=20, prior_mode="auto", default_heads=None):
    if prior_mode not in {"auto", "score", "rank", "uniform"}:
        raise ValueError(f"Unknown prior_mode={prior_mode}")

    if not path:
        heads = (default_heads or DEFAULT_LLAVA15_HEADS)[:top_k]
        if prior_mode == "uniform":
            return heads, uniform_priors(heads), "uniform_default"
        return heads, rank_priors(heads), "rank_default"

    with open(path, "r") as f:
        data = json.load(f)
    heads = data.get("hal_heads", default_heads or DEFAULT_LLAVA15_HEADS)[:top_k]

    if prior_mode == "uniform":
        return heads, uniform_priors(heads), "uniform"
    if prior_mode == "rank":
        return heads, rank_priors(heads), "rank_forced"

    score_items = []
    score_source = None
    for field in ("contrastive_scores", "hal_head_scores"):
        if field in data:
            score_items = _score_items_from_field(data[field])
            score_source = field
            break

    if score_items:
        allowed = {head_key(*head) for head in heads}
        priors = {key: value for key, value in score_priors(score_items).items() if key in allowed}
        for key, value in rank_priors(heads).items():
            priors.setdefault(key, value)
        return heads, priors, f"score_{score_source}"

    if prior_mode == "score":
        return heads, rank_priors(heads), "rank_fallback_no_scores"
    return heads, rank_priors(heads), "rank_fallback"


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q / 100.0))
    return values[idx]


def headwise_percentile_thresholds(head_records, q_low=60, q_high=90, fallback_low=0.4, fallback_high=0.9):
    thresholds = {}
    for key, values in head_records.items():
        low = percentile(values, q_low)
        high = percentile(values, q_high)
        if low is None:
            low = fallback_low
        if high is None or high <= low:
            high = max(fallback_high, low + 1e-4)
        thresholds[key] = {"low": float(low), "high": float(high)}
    return thresholds


def set_llava_head_config(model, model_path, attribution_path=None, top_k=20):
    heads, priors, source = load_head_priors(
        attribution_path,
        top_k=top_k,
        default_heads=default_heads_for_model(model_path),
    )
    if model_path == "liuhaotian/llava-v1.5-7b":
        model.config.img_start_pos = 35
        model.config.img_length = 576
    elif model_path == "liuhaotian/llava-v1.5-13b":
        model.config.img_start_pos = 35
        model.config.img_length = 576
    elif model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        raise ValueError(f"No built-in image-token span for model_path={model_path}")
    model.config.hal_attention_heads = heads
    model.config.head_attribution_priors = priors
    model.config.head_attribution_prior_source = source
    return heads, priors, source
