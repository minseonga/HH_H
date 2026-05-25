import os
import json
import argparse
import uuid
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import torch
from torch.utils.data import Dataset, DataLoader

import math
import shutil
import random
import numpy as np
from PIL import Image
from transformers import set_seed
from pycocotools.coco import COCO
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, load_head_priors

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_int_ranges(spec):
    values = []
    if not spec:
        return values
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return sorted(set(values))

def load_first_subtoken_ids(tokenizer, vocab_path):
    token_ids = []
    seen = set()
    if not vocab_path:
        return token_ids
    if not os.path.exists(vocab_path):
        print(f"[warn] missing object vocab path: {vocab_path}")
        return token_ids
    with open(vocab_path) as f:
        for line in f:
            for word in line.strip().split(","):
                word = word.strip()
                if not word:
                    continue
                ids = tokenizer(word, add_special_tokens=False).get("input_ids", [])
                if not ids:
                    continue
                token_id = int(ids[0])
                if token_id in seen:
                    continue
                seen.add(token_id)
                token_ids.append(token_id)
    return token_ids


def load_query_direction_gate(calibration_npz, top_k=0, min_auroc=0.0):
    data = np.load(calibration_npz)
    layers = data["layers"].astype(int)
    heads = data["heads"].astype(int)
    directions = data["directions"].astype(np.float32)
    thresholds = data["threshold_midpoint"].astype(np.float32)
    test_auroc = data["test_auroc"].astype(np.float32)

    candidates = []
    for idx, auc in enumerate(test_auroc.tolist()):
        if float(auc) < float(min_auroc):
            continue
        candidates.append((float(auc), idx))
    candidates.sort(reverse=True)
    if top_k > 0:
        candidates = candidates[:top_k]

    direction_dict = {}
    threshold_dict = {}
    rows = []
    for rank, (_, idx) in enumerate(candidates, start=1):
        layer = int(layers[idx])
        head = int(heads[idx])
        key = f"{layer}:{head}"
        direction_dict[key] = torch.from_numpy(directions[idx].copy())
        threshold_dict[key] = float(thresholds[idx])
        rows.append({
            "rank": rank,
            "head_key": key,
            "test_auroc": float(test_auroc[idx]),
            "threshold": float(thresholds[idx]),
        })
    return direction_dict, threshold_dict, rows


def _top2_margin(logits):
    top_vals, top_ids = torch.topk(logits, k=2)
    return top_vals, top_ids, top_vals[0] - top_vals[1]


@torch.inference_mode()
def build_layer_contrastive_records_from_hidden_states(model, output_dict, args):
    hidden_states_by_step = output_dict.get("hidden_states", None)
    if not hidden_states_by_step:
        return []

    layers = parse_int_ranges(args.layer_contrastive_layers) or [args.layer_contrastive_layer]
    lm_head = model.get_output_embeddings()
    records = []
    eps = 1e-8
    for step_idx, step_hidden_states in enumerate(hidden_states_by_step):
        if not step_hidden_states:
            continue
        num_hidden_states = len(step_hidden_states)
        valid_layers = []
        for layer in layers:
            layer = int(layer)
            if layer < 0:
                layer += num_hidden_states
            if 0 <= layer < num_hidden_states:
                valid_layers.append(layer)
        if not valid_layers:
            continue

        final_vec = step_hidden_states[-1][0, -1, :]
        final_logits = lm_head(final_vec).float()
        mid_logits = None
        for layer in valid_layers:
            layer_logits = lm_head(step_hidden_states[layer][0, -1, :]).float()
            mid_logits = layer_logits if mid_logits is None else mid_logits + layer_logits
        mid_logits = mid_logits / float(len(valid_layers))

        final_log_probs = torch.log_softmax(final_logits, dim=-1)
        mid_log_probs = torch.log_softmax(mid_logits, dim=-1)
        final_probs = final_log_probs.exp()
        mid_probs = mid_log_probs.exp()
        final_entropy = -torch.sum(final_probs * final_log_probs)
        final_entropy_norm = torch.clamp(
            final_entropy / math.log(float(final_logits.shape[-1])),
            min=0.0,
            max=1.0,
        )
        mixture_probs = 0.5 * (final_probs + mid_probs)
        mixture_log_probs = torch.log(torch.clamp(mixture_probs, min=eps))
        kl_final_mid = torch.sum(final_probs * (final_log_probs - mid_log_probs))
        kl_mid_final = torch.sum(mid_probs * (mid_log_probs - final_log_probs))
        js = 0.5 * (
            torch.sum(final_probs * (final_log_probs - mixture_log_probs))
            + torch.sum(mid_probs * (mid_log_probs - mixture_log_probs))
        )
        js_norm = torch.clamp(js / math.log(2.0), min=0.0, max=1.0)

        final_top_vals, final_top_ids, final_margin = _top2_margin(final_logits)
        mid_top_vals, mid_top_ids, mid_margin = _top2_margin(mid_logits)
        phase = "prefill" if step_idx == 0 else "decode"
        records.append({
            "record_type": "layer_contrastive",
            "phase": phase,
            "call_index": int(step_idx),
            "forward_index": int(step_idx),
            "step_index": int(step_idx),
            "token_pos": int(step_idx),
            "position": -1,
            "layers": ",".join(str(layer) for layer in valid_layers),
            "alpha": float(args.layer_contrastive_alpha),
            "gate_feature": args.layer_contrastive_gate_feature,
            "gate": float(js_norm.detach().cpu().item()),
            "kl_final_mid": float(kl_final_mid.detach().cpu().item()),
            "kl_mid_final": float(kl_mid_final.detach().cpu().item()),
            "sym_kl": float((0.5 * (kl_final_mid + kl_mid_final)).detach().cpu().item()),
            "js_divergence_norm": float(js_norm.detach().cpu().item()),
            "final_entropy": float(final_entropy.detach().cpu().item()),
            "final_entropy_norm": float(final_entropy_norm.detach().cpu().item()),
            "final_top1_token_id": int(final_top_ids[0].detach().cpu().item()),
            "mid_top1_token_id": int(mid_top_ids[0].detach().cpu().item()),
            "corrected_top1_token_id": int(final_top_ids[0].detach().cpu().item()),
            "final_top1_logit": float(final_top_vals[0].detach().cpu().item()),
            "mid_top1_logit": float(mid_top_vals[0].detach().cpu().item()),
            "corrected_top1_logit": float(final_top_vals[0].detach().cpu().item()),
            "final_top1_top2_margin": float(final_margin.detach().cpu().item()),
            "mid_top1_top2_margin": float(mid_margin.detach().cpu().item()),
            "corrected_top1_top2_margin": float(final_margin.detach().cpu().item()),
            "final_mid_top1_agree": bool(final_top_ids[0].detach().cpu().item() == mid_top_ids[0].detach().cpu().item()),
            "final_corrected_top1_agree": True,
        })
    return records

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.dataset == 'coco':
        caption_file_path = args.caption_file_path
        coco = COCO(caption_file_path)
        img_ids = coco.getImgIds()
        sampled_img_ids = random.sample(img_ids, args.num_samples)

        questions = []
        dest_image_folder = os.path.join(os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0], 'images', f'seed{args.seed}_{args.num_samples}')
        os.makedirs(dest_image_folder, exist_ok=True)
        for sampled_img_id in sampled_img_ids:
            image_file = coco.loadImgs(sampled_img_id)[0]["file_name"]
            question = {
                "question_id": sampled_img_id,
                "image": image_file,
                "text": "Please describe this image in detail.",
            }
            shutil.copyfile(os.path.join(args.image_folder, image_file), os.path.join(dest_image_folder, image_file))
            questions.append(question)

    elif args.dataset == 'nocaps':  
        caption_file_path = args.caption_file_path
        val_caps = json.load(open(caption_file_path))
        image_infos = val_caps["images"]
        out_image_infos = [image_info for image_info in image_infos if image_info['domain'] == 'out-domain']
        sampled_img_infos = random.sample(out_image_infos, args.num_samples)

        questions = []
        dest_image_folder = os.path.join(os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0], 'images', f'seed{args.seed}_{args.num_samples}')
        os.makedirs(dest_image_folder, exist_ok=True)
        for sampled_img_info in sampled_img_infos:
            image_file = sampled_img_info['file_name']
            image_id = sampled_img_info['id']
            question = {
                "question_id": sampled_img_info['id'],
                "image": sampled_img_info['file_name'],
                "text": "Please describe this image in detail.",
            }
            shutil.copyfile(os.path.join(args.image_folder, image_file), os.path.join(dest_image_folder, f'{image_id}_{image_file}'))
            questions.append(question)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    if (
        args.adaptive_deactivate
        or args.soft_deactivate
        or args.dynamic_deactivate
        or args.attribution_soft_deactivate
        or args.retention_aware_deactivate
        or args.visual_gate_deactivate
        or args.wide_gate_deactivate
        or args.online_value_selector_deactivate
        or args.unsupported_component_deactivate
    ):
        if model_path == 'liuhaotian/llava-v1.5-7b':
            model.config.img_start_pos = 35
            model.config.img_length = 576
        elif model_path == 'liuhaotian/llava-v1.5-13b':
            model.config.img_start_pos = 35
            model.config.img_length = 576

        elif model_path == 'liuhaotian/llava-v1.6-34b':
            model.config.img_start_pos = 33
            model.config.img_length = 1948

        heads, priors, prior_source = load_head_priors(
            args.attention_head_path,
            top_k=args.top_k,
            prior_mode=args.head_prior_mode,
            default_heads=default_heads_for_model(model_path),
        )
        model.config.hal_attention_heads = heads
        model.config.head_attribution_priors = priors
        model.config.head_attribution_prior_source = prior_source

        model.config.adhh_threshold = args.adhh_threshold
        if args.adaptive_deactivate:
            model.config.adaptive_deactivate = True
        if args.soft_deactivate:
            model.config.soft_deactivate = True
            model.config.soft_gamma = args.soft_gamma
            model.config.soft_temperature = args.soft_temperature
        if args.dynamic_deactivate:
            model.config.dynamic_deactivate = True
            model.config.dynamic_gamma = args.dynamic_gamma
            model.config.dynamic_temperature = args.dynamic_temperature
            model.config.dynamic_margin_weight = args.dynamic_margin_weight
            model.config.dynamic_ratio_weight = args.dynamic_ratio_weight
            model.config.dynamic_consensus_weight = args.dynamic_consensus_weight
            model.config.dynamic_bias = args.dynamic_bias
        if args.attribution_soft_deactivate:
            model.config.attribution_soft_deactivate = True
            model.config.attribution_soft_gamma = args.attribution_soft_gamma
            model.config.attribution_soft_mode = args.attribution_soft_mode
            model.config.attribution_tau_low = args.attribution_tau_low
            model.config.attribution_tau_high = args.attribution_tau_high
            if args.head_thresholds_path:
                with open(args.head_thresholds_path, "r") as f:
                    threshold_data = json.load(f)
                model.config.head_text_thresholds = threshold_data.get("head_text_thresholds", threshold_data)
        if args.retention_aware_deactivate:
            model.config.retention_aware_deactivate = True
            model.config.retention_policy_mode = args.retention_policy_mode
            model.config.retention_feature = args.retention_feature
            model.config.retention_rho = args.retention_rho
            model.config.retention_lambda = args.retention_lambda
            model.config.retention_soft_gamma = args.retention_soft_gamma
            model.config.retention_soft_temperature = args.retention_soft_temperature
        if args.visual_gate_deactivate:
            model.config.visual_gate_deactivate = True
            model.config.visual_gate_gamma = args.visual_gate_gamma
            model.config.visual_gate_beta = args.visual_gate_beta
            model.config.visual_gate_v0 = args.visual_gate_v0
            model.config.visual_gate_temperature = args.visual_gate_temperature
            model.config.visual_gate_proxy = args.visual_gate_proxy
            model.config.visual_gate_recent_weight = args.visual_gate_recent_weight
            model.config.visual_gate_recent_window = args.visual_gate_recent_window
            model.config.visual_gate_tau_low = args.visual_gate_tau_low
            model.config.visual_gate_tau_high = args.visual_gate_tau_high
            if args.head_thresholds_path:
                with open(args.head_thresholds_path, "r") as f:
                    threshold_data = json.load(f)
                model.config.head_text_thresholds = threshold_data.get("head_text_thresholds", threshold_data)
        if args.wide_gate_deactivate:
            model.config.wide_gate_deactivate = True
            model.config.wide_gate_mode = args.wide_gate_mode
            model.config.wide_gate_feature = args.wide_gate_feature
            model.config.wide_gate_text_tau = args.wide_gate_text_tau
            model.config.wide_gate_text_high = args.wide_gate_text_high
            model.config.wide_gate_gamma = args.wide_gate_gamma
            model.config.wide_gate_norm_threshold = args.wide_gate_norm_threshold
            model.config.wide_gate_norm_low = args.wide_gate_norm_low
            model.config.wide_gate_norm_high = args.wide_gate_norm_high
            model.config.wide_gate_norm_source = args.wide_gate_norm_source
            if args.head_norm_thresholds_path:
                with open(args.head_norm_thresholds_path, "r") as f:
                    threshold_data = json.load(f)
                model.config.head_norm_thresholds = threshold_data.get("head_norm_thresholds", threshold_data)
        if args.online_value_selector_deactivate:
            model.config.online_value_selector_deactivate = True
            model.config.online_value_selector_mode = args.online_value_selector_mode
            model.config.online_value_selector_text_tau = args.online_value_selector_text_tau
            model.config.online_value_selector_gamma = args.online_value_selector_gamma
            model.config.online_value_selector_layer_top_k = args.online_value_selector_layer_top_k
            model.config.online_value_selector_require_text_trigger = not args.online_value_selector_no_text_trigger
            model.config.online_value_selector_soft_threshold = args.online_value_selector_soft_threshold
            model.config.online_value_selector_hard_threshold = args.online_value_selector_hard_threshold
            model.config.online_value_selector_norm_threshold = args.online_value_selector_norm_threshold
            model.config.online_value_selector_norm_low = args.online_value_selector_norm_low
            model.config.online_value_selector_norm_high = args.online_value_selector_norm_high
            model.config.online_value_selector_norm_source = args.online_value_selector_norm_source
            if args.head_norm_thresholds_path:
                with open(args.head_norm_thresholds_path, "r") as f:
                    threshold_data = json.load(f)
                model.config.head_norm_thresholds = threshold_data.get("head_norm_thresholds", threshold_data)
        if args.layer_contrastive_deactivate:
            model.config.layer_contrastive_deactivate = True
            model.config.layer_contrastive_layers = parse_int_ranges(args.layer_contrastive_layers)
            if not model.config.layer_contrastive_layers:
                model.config.layer_contrastive_layers = [args.layer_contrastive_layer]
            model.config.layer_contrastive_alpha = args.layer_contrastive_alpha
            model.config.layer_contrastive_gate_feature = args.layer_contrastive_gate_feature
            model.config.layer_contrastive_gate_power = args.layer_contrastive_gate_power
            model.config.layer_contrastive_margin_temperature = args.layer_contrastive_margin_temperature
            model.config.layer_contrastive_phase = args.layer_contrastive_phase
            model.config.record_layer_contrastive_diagnostics = args.record_layer_contrastive_diagnostics
            model.config.layer_contrastive_diagnostics_max_records = args.layer_contrastive_diagnostics_max_records
        if args.unsupported_component_deactivate:
            model.config.unsupported_component_deactivate = True
            model.config.unsupported_component_mode = args.unsupported_component_mode
            model.config.unsupported_component_layer_top_k = args.unsupported_component_layer_top_k
            model.config.unsupported_component_gamma = args.unsupported_component_gamma
            model.config.unsupported_component_action = args.unsupported_component_action
            model.config.unsupported_component_delta_budget = args.unsupported_component_delta_budget
            model.config.unsupported_component_unsupported_weight = args.unsupported_component_unsupported_weight
            model.config.unsupported_component_image_weight = args.unsupported_component_image_weight
            model.config.unsupported_component_soft_threshold = args.unsupported_component_soft_threshold
            model.config.unsupported_component_hard_threshold = args.unsupported_component_hard_threshold
            model.config.unsupported_component_risk_feature = args.unsupported_component_risk_feature
            model.config.unsupported_component_score_norm = args.unsupported_component_score_norm
            model.config.unsupported_component_score_low = args.unsupported_component_score_low
            model.config.unsupported_component_score_high = args.unsupported_component_score_high
            model.config.unsupported_component_phase = args.unsupported_component_phase
            model.config.unsupported_component_layers = parse_int_ranges(args.unsupported_component_layers)
            model.config.unsupported_component_prefill_protect_top_k = args.unsupported_component_prefill_protect_top_k
            model.config.unsupported_component_recent_text_window = args.unsupported_component_recent_text_window
            model.config.unsupported_component_sink_top_k = args.unsupported_component_sink_top_k
            model.config.unsupported_component_sink_offsets = parse_int_ranges(args.unsupported_component_sink_offsets)
            if args.unsupported_component_risk_feature in (
                "unsupported_object_logit",
                "text_mass_x_object_logit_disagreement",
            ):
                object_token_ids = load_first_subtoken_ids(tokenizer, args.unsupported_component_object_vocab_path)
                if object_token_ids:
                    lm_head = model.get_output_embeddings().weight
                    object_token_tensor = torch.tensor(object_token_ids, dtype=torch.long, device=lm_head.device)
                    model.config.unsupported_component_object_lm_head = lm_head.index_select(0, object_token_tensor).detach()
                    print(
                        f"[info] unsupported object-logit vocab tokens: {len(object_token_ids)} "
                        f"from {args.unsupported_component_object_vocab_path}"
                    )
                else:
                    print("[warn] object-logit risk requested but no object token ids were loaded")
            model.config.unsupported_component_all_heads = args.unsupported_component_all_heads
            model.config.record_unsupported_component_diagnostics = args.record_unsupported_component_diagnostics
            model.config.record_unsupported_component_candidates = args.record_unsupported_component_candidates
            model.config.unsupported_component_diagnostics_max_records = args.unsupported_component_diagnostics_max_records
            model.config.unsupported_component_query_gate_mode = args.unsupported_component_query_gate_mode
            model.config.unsupported_component_query_gate_aggregation = args.unsupported_component_query_gate_aggregation
            model.config.unsupported_component_query_gate_temperature = args.unsupported_component_query_gate_temperature
            model.config.unsupported_component_query_gate_default = args.unsupported_component_query_gate_default
            model.config.unsupported_component_query_gate_min = args.unsupported_component_query_gate_min
            model.config.unsupported_component_query_gate_power = args.unsupported_component_query_gate_power
            model.config.unsupported_component_query_gate_state = {}
            if args.unsupported_component_query_gate_calibration:
                if not os.path.exists(args.unsupported_component_query_gate_calibration):
                    raise FileNotFoundError(args.unsupported_component_query_gate_calibration)
                directions, thresholds, direction_rows = load_query_direction_gate(
                    args.unsupported_component_query_gate_calibration,
                    top_k=args.unsupported_component_query_gate_top_k,
                    min_auroc=args.unsupported_component_query_gate_min_auroc,
                )
                model.config.unsupported_component_query_gate_directions = directions
                model.config.unsupported_component_query_gate_thresholds = thresholds
                print(
                    f"[info] unsupported query gate directions: {len(directions)} "
                    f"from {args.unsupported_component_query_gate_calibration}"
                )
                for row in direction_rows[:10]:
                    print(
                        "[info] query gate direction "
                        f"rank={row['rank']} head={row['head_key']} "
                        f"auc={row['test_auroc']:.4f} threshold={row['threshold']:.4f}"
                    )
            else:
                model.config.unsupported_component_query_gate_directions = {}
                model.config.unsupported_component_query_gate_thresholds = {}

    if args.query_direction_project:
        if not args.query_direction_calibration:
            raise ValueError("--query_direction_calibration is required with --query_direction_project")
        if not os.path.exists(args.query_direction_calibration):
            raise FileNotFoundError(args.query_direction_calibration)
        directions, thresholds, direction_rows = load_query_direction_gate(
            args.query_direction_calibration,
            top_k=args.query_direction_top_k,
            min_auroc=args.query_direction_min_auroc,
        )
        if not directions:
            raise ValueError(
                "No query directions selected; lower --query_direction_min_auroc "
                "or increase --query_direction_top_k."
            )
        model.config.query_direction_project = True
        model.config.query_direction_directions = directions
        model.config.query_direction_thresholds = thresholds
        model.config.query_direction_strength = args.query_direction_strength
        model.config.query_direction_gate_mode = args.query_direction_gate_mode
        model.config.query_direction_temperature = args.query_direction_temperature
        model.config.query_direction_positive_only = not args.query_direction_allow_negative
        model.config.query_direction_phase = args.query_direction_phase
        model.config.record_query_projection_diagnostics = args.record_query_projection_diagnostics
        model.config.query_projection_diagnostics = []
        print(
            f"[info] query projection directions: {len(directions)} "
            f"from {args.query_direction_calibration}"
        )
        print(
            f"[info] query projection: strength={args.query_direction_strength} "
            f"phase={args.query_direction_phase} gate={args.query_direction_gate_mode} "
            f"positive_only={not args.query_direction_allow_negative}"
        )
        for row in direction_rows[:10]:
            print(
                "[info] query projection direction "
                f"rank={row['rank']} head={row['head_key']} "
                f"auc={row['test_auroc']:.4f} threshold={row['threshold']:.4f}"
            )
    else:
        model.config.query_direction_project = False

    unsupported_diag_file = None
    if args.record_unsupported_component_diagnostics:
        unsupported_diag_path = args.unsupported_component_diagnostics_file
        if not unsupported_diag_path:
            unsupported_diag_path = os.path.join(
                os.path.dirname(answers_file),
                "unsupported_component_diagnostics.jsonl",
            )
        unsupported_diag_dir = os.path.dirname(unsupported_diag_path)
        if unsupported_diag_dir:
            os.makedirs(unsupported_diag_dir, exist_ok=True)
        unsupported_diag_file = open(unsupported_diag_path, "w")
        print(f"[info] unsupported component diagnostics: {unsupported_diag_path}")

    layer_contrastive_diag_file = None
    if args.record_layer_contrastive_diagnostics:
        layer_contrastive_diag_path = args.layer_contrastive_diagnostics_file
        if not layer_contrastive_diag_path:
            layer_contrastive_diag_path = os.path.join(
                os.path.dirname(answers_file),
                "layer_contrastive_diagnostics.jsonl",
            )
        layer_contrastive_diag_dir = os.path.dirname(layer_contrastive_diag_path)
        if layer_contrastive_diag_dir:
            os.makedirs(layer_contrastive_diag_dir, exist_ok=True)
        layer_contrastive_diag_file = open(layer_contrastive_diag_path, "w")
        print(f"[info] layer contrastive diagnostics: {layer_contrastive_diag_path}")

    query_projection_diag_file = None
    if args.record_query_projection_diagnostics:
        query_projection_diag_path = args.query_projection_diagnostics_file
        if not query_projection_diag_path:
            query_projection_diag_path = os.path.join(
                os.path.dirname(answers_file),
                "query_projection_diagnostics.jsonl",
            )
        query_projection_diag_dir = os.path.dirname(query_projection_diag_path)
        if query_projection_diag_dir:
            os.makedirs(query_projection_diag_dir, exist_ok=True)
        query_projection_diag_file = open(query_projection_diag_path, "w")
        print(f"[info] query projection diagnostics: {query_projection_diag_path}")

    count = 0
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        count += 1
        question_id = line["question_id"]
        cur_prompt = line["text"]
        image_file = line["image"]

        if args.record_unsupported_component_diagnostics:
            model.config.unsupported_component_diagnostics = []
            model.config.unsupported_component_call_index = 0
        if args.unsupported_component_deactivate:
            model.config.unsupported_component_prefill_protect_heads = {}
            model.config.unsupported_component_query_gate_state = {}
        if args.record_layer_contrastive_diagnostics:
            model.config.layer_contrastive_diagnostics = []
            model.config.layer_contrastive_call_index = 0
            model.config.layer_contrastive_forward_index = 0
        if args.record_query_projection_diagnostics:
            model.config.query_projection_diagnostics = []

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_dict = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=args.record_layer_contrastive_diagnostics,
                output_scores=args.record_token_score_diagnostics,
                return_dict_in_generate=True)

        output_ids = output_dict['sequences']
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(question_id, outputs)

        ans_id = uuid.uuid4().hex
        ans_file.write(json.dumps({"question_id": question_id,
                                "image": image_file,
                                "prompt": cur_prompt,
                                "text": outputs,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "metadata": {}}) + "\n")
        ans_file.flush()

        if unsupported_diag_file is not None:
            if args.record_token_score_diagnostics:
                score_tensors = output_dict.get("scores", None)
                if score_tensors is not None:
                    for token_pos, score_tensor in enumerate(score_tensors):
                        score = score_tensor[0].detach().float()
                        probs = torch.softmax(score, dim=-1)
                        log_probs = torch.log_softmax(score, dim=-1)
                        entropy = -(probs * log_probs).sum()
                        top_vals, top_ids = torch.topk(score, k=min(5, score.shape[-1]), largest=True)
                        top_ids_list = [int(item) for item in top_ids.detach().cpu().tolist()]
                        top_vals_list = [float(item) for item in top_vals.detach().float().cpu().tolist()]
                        token_record = {
                            "record_type": "token_score",
                            "phase": "decode",
                            "question_id": question_id,
                            "image": image_file,
                            "token_pos": int(token_pos),
                            "step_index": int(token_pos),
                            "logit_entropy": float(entropy.detach().float().cpu().item()),
                            "top1_top2_margin": (
                                float(top_vals_list[0] - top_vals_list[1])
                                if len(top_vals_list) >= 2 else None
                            ),
                            "negative_top1_top2_margin": (
                                float(top_vals_list[1] - top_vals_list[0])
                                if len(top_vals_list) >= 2 else None
                            ),
                            "top1_logit": top_vals_list[0] if top_vals_list else None,
                            "top2_logit": top_vals_list[1] if len(top_vals_list) >= 2 else None,
                            "top1_token_id": top_ids_list[0] if top_ids_list else None,
                            "top1_token": tokenizer.decode([top_ids_list[0]]) if top_ids_list else "",
                            "top5_token_ids": ",".join(str(item) for item in top_ids_list),
                            "top5_logits": ",".join(f"{item:.6g}" for item in top_vals_list),
                            "top5_tokens": "|".join(tokenizer.decode([item]).replace("\n", "\\n") for item in top_ids_list),
                        }
                        unsupported_diag_file.write(json.dumps(token_record) + "\n")
            for record in getattr(model.config, "unsupported_component_diagnostics", []):
                record = dict(record)
                record["question_id"] = question_id
                record["image"] = image_file
                if record.get("record_type") != "candidate_head":
                    record["caption"] = outputs
                unsupported_diag_file.write(json.dumps(record) + "\n")
            unsupported_diag_file.flush()

        if layer_contrastive_diag_file is not None:
            layer_contrastive_records = getattr(model.config, "layer_contrastive_diagnostics", [])
            if not layer_contrastive_records:
                layer_contrastive_records = build_layer_contrastive_records_from_hidden_states(
                    model,
                    output_dict,
                    args,
                )
            for record in layer_contrastive_records:
                record = dict(record)
                record["question_id"] = question_id
                record["image"] = image_file
                record["caption"] = outputs
                layer_contrastive_diag_file.write(json.dumps(record) + "\n")
            layer_contrastive_diag_file.flush()

        if query_projection_diag_file is not None:
            for record in getattr(model.config, "query_projection_diagnostics", []):
                record = dict(record)
                record["question_id"] = question_id
                record["image"] = image_file
                record["caption"] = outputs
                query_projection_diag_file.write(json.dumps(record) + "\n")
            query_projection_diag_file.flush()

    if unsupported_diag_file is not None:
        unsupported_diag_file.close()
    if layer_contrastive_diag_file is not None:
        layer_contrastive_diag_file.close()
    if query_projection_diag_file is not None:
        query_projection_diag_file.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--caption_file_path", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive_deactivate", action='store_true', default=False)
    parser.add_argument("--soft_deactivate", action='store_true', default=False)
    parser.add_argument("--dynamic_deactivate", action='store_true', default=False)
    parser.add_argument("--attribution_soft_deactivate", action='store_true', default=False)
    parser.add_argument("--retention_aware_deactivate", action='store_true', default=False)
    parser.add_argument("--visual_gate_deactivate", action='store_true', default=False)
    parser.add_argument("--wide_gate_deactivate", action='store_true', default=False)
    parser.add_argument("--online_value_selector_deactivate", action='store_true', default=False)
    parser.add_argument("--layer_contrastive_deactivate", action="store_true", default=False)
    parser.add_argument("--unsupported_component_deactivate", action='store_true', default=False)
    parser.add_argument("--adhh_threshold", type=float, default=0.0)
    parser.add_argument("--attention_head_path", type=str, default="")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--head_prior_mode", type=str, default="auto", choices=["auto", "score", "rank", "uniform"])
    parser.add_argument("--soft_gamma", type=float, default=0.5)
    parser.add_argument("--soft_temperature", type=float, default=0.05)
    parser.add_argument("--dynamic_gamma", type=float, default=1.0)
    parser.add_argument("--dynamic_temperature", type=float, default=0.05)
    parser.add_argument("--dynamic_margin_weight", type=float, default=1.0)
    parser.add_argument("--dynamic_ratio_weight", type=float, default=0.25)
    parser.add_argument("--dynamic_consensus_weight", type=float, default=0.5)
    parser.add_argument("--dynamic_bias", type=float, default=0.0)
    parser.add_argument("--attribution_soft_gamma", type=float, default=1.0)
    parser.add_argument("--attribution_soft_mode", type=str, default="linear", choices=["linear", "sqrt", "quadratic", "budget"])
    parser.add_argument("--attribution_tau_low", type=float, default=0.4)
    parser.add_argument("--attribution_tau_high", type=float, default=0.9)
    parser.add_argument("--head_thresholds_path", type=str, default="")
    parser.add_argument("--retention_policy_mode", type=str, default="hard_or_soft", choices=["hard_or_soft", "cap"])
    parser.add_argument(
        "--retention_feature",
        type=str,
        default="mean_prior_text_mass",
        choices=["mean_prior_text_mass", "mean_excess", "weighted_excess", "trigger_frac", "weighted_trigger_count"],
    )
    parser.add_argument("--retention_rho", type=float, default=0.1)
    parser.add_argument("--retention_lambda", type=float, default=1.0)
    parser.add_argument("--retention_soft_gamma", type=float, default=0.75)
    parser.add_argument("--retention_soft_temperature", type=float, default=0.05)
    parser.add_argument("--visual_gate_gamma", type=float, default=1.0)
    parser.add_argument("--visual_gate_beta", type=float, default=0.75)
    parser.add_argument("--visual_gate_v0", type=float, default=0.5)
    parser.add_argument("--visual_gate_temperature", type=float, default=0.15)
    parser.add_argument("--visual_gate_proxy", type=str, default="value", choices=["value", "mass", "value_recent"])
    parser.add_argument("--visual_gate_recent_weight", type=float, default=0.0)
    parser.add_argument("--visual_gate_recent_window", type=int, default=16)
    parser.add_argument("--visual_gate_tau_low", type=float, default=0.4)
    parser.add_argument("--visual_gate_tau_high", type=float, default=0.9)
    parser.add_argument("--wide_gate_mode", type=str, default="hard", choices=["hard", "continuous"])
    parser.add_argument("--wide_gate_feature", type=str, default="text_norm", choices=["text", "norm", "text_norm"])
    parser.add_argument("--wide_gate_text_tau", type=float, default=0.4)
    parser.add_argument("--wide_gate_text_high", type=float, default=0.9)
    parser.add_argument("--wide_gate_gamma", type=float, default=1.0)
    parser.add_argument("--wide_gate_norm_threshold", type=float, default=0.0)
    parser.add_argument("--wide_gate_norm_low", type=float, default=0.0)
    parser.add_argument("--wide_gate_norm_high", type=float, default=1.0)
    parser.add_argument("--wide_gate_norm_source", type=str, default="text_value", choices=["text_value", "head_output"])
    parser.add_argument("--online_value_selector_mode", type=str, default="continuous", choices=["hard", "continuous", "hybrid"])
    parser.add_argument("--online_value_selector_layer_top_k", type=int, default=1)
    parser.add_argument("--online_value_selector_text_tau", type=float, default=0.4)
    parser.add_argument("--online_value_selector_gamma", type=float, default=1.0)
    parser.add_argument("--online_value_selector_soft_threshold", type=float, default=0.25)
    parser.add_argument("--online_value_selector_hard_threshold", type=float, default=0.75)
    parser.add_argument("--online_value_selector_norm_threshold", type=float, default=0.0)
    parser.add_argument("--online_value_selector_norm_low", type=float, default=0.0)
    parser.add_argument("--online_value_selector_norm_high", type=float, default=1.0)
    parser.add_argument("--online_value_selector_norm_source", type=str, default="text_value", choices=["text_value", "head_output"])
    parser.add_argument("--online_value_selector_no_text_trigger", action="store_true", default=False)
    parser.add_argument("--layer_contrastive_layer", type=int, default=16)
    parser.add_argument("--layer_contrastive_layers", type=str, default="")
    parser.add_argument("--layer_contrastive_alpha", type=float, default=0.5)
    parser.add_argument(
        "--layer_contrastive_gate_feature",
        type=str,
        default="js_divergence",
        choices=["constant", "js_divergence", "final_entropy", "low_margin", "js_x_entropy", "js_x_low_margin"],
    )
    parser.add_argument("--layer_contrastive_gate_power", type=float, default=1.0)
    parser.add_argument("--layer_contrastive_margin_temperature", type=float, default=1.0)
    parser.add_argument("--layer_contrastive_phase", type=str, default="decode", choices=["all", "prefill", "decode"])
    parser.add_argument("--record_layer_contrastive_diagnostics", action="store_true", default=False)
    parser.add_argument("--layer_contrastive_diagnostics_file", type=str, default="")
    parser.add_argument("--layer_contrastive_diagnostics_max_records", type=int, default=0)
    parser.add_argument("--query_direction_project", action="store_true", default=False)
    parser.add_argument("--query_direction_calibration", type=str, default="")
    parser.add_argument("--query_direction_top_k", type=int, default=1)
    parser.add_argument("--query_direction_min_auroc", type=float, default=0.0)
    parser.add_argument("--query_direction_strength", type=float, default=0.5)
    parser.add_argument(
        "--query_direction_gate_mode",
        type=str,
        default="none",
        choices=["none", "positive", "threshold", "sigmoid"],
    )
    parser.add_argument("--query_direction_temperature", type=float, default=0.05)
    parser.add_argument("--query_direction_allow_negative", action="store_true", default=False)
    parser.add_argument("--query_direction_phase", type=str, default="decode", choices=["all", "prefill", "decode"])
    parser.add_argument("--record_query_projection_diagnostics", action="store_true", default=False)
    parser.add_argument("--query_projection_diagnostics_file", type=str, default="")
    parser.add_argument("--unsupported_component_mode", type=str, default="continuous", choices=["hard", "continuous", "hybrid"])
    parser.add_argument("--unsupported_component_layer_top_k", type=int, default=1)
    parser.add_argument("--unsupported_component_gamma", type=float, default=0.5)
    parser.add_argument(
        "--unsupported_component_action",
        type=str,
        default="suppress_unsupported",
        choices=[
            "suppress_unsupported",
            "boost_image",
            "boost_image_geomean",
            "boost_image_matched",
            "prefill_balanced_steer",
            "scale_head_output",
            "scale_text_component",
        ],
    )
    parser.add_argument("--unsupported_component_delta_budget", type=float, default=0.0)
    parser.add_argument("--unsupported_component_unsupported_weight", type=float, default=1.0)
    parser.add_argument("--unsupported_component_image_weight", type=float, default=1.0)
    parser.add_argument("--unsupported_component_soft_threshold", type=float, default=0.25)
    parser.add_argument("--unsupported_component_hard_threshold", type=float, default=0.75)
    parser.add_argument(
        "--unsupported_component_score_norm",
        type=str,
        default="candidate_minmax",
        choices=["candidate_minmax", "candidate_max", "candidate_sum", "absolute", "identity"],
    )
    parser.add_argument("--unsupported_component_score_low", type=float, default=0.0)
    parser.add_argument("--unsupported_component_score_high", type=float, default=1.0)
    parser.add_argument(
        "--unsupported_component_phase",
        type=str,
        default="all",
        choices=["all", "prefill", "decode"],
    )
    parser.add_argument("--unsupported_component_layers", type=str, default="")
    parser.add_argument(
        "--unsupported_component_risk_feature",
        type=str,
        default="unsupported_norm_x_low_anchor",
        choices=[
            "unsupported_norm",
            "unsupported_total_ratio",
            "low_img_mass",
            "semantic_img_mass",
            "semantic_low_img_mass",
            "text_mass_x_disagreement",
            "text_mass_x_max_peer_disagreement",
            "text_mass_x_object_logit_disagreement",
            "unsupported_norm_x_low_anchor",
            "unsupported_total_ratio_x_low_anchor",
            "unsupported_norm_x_low_visual",
            "unsupported_norm_x_low_anchor_x_low_visual",
            "unsupported_head_ratio",
            "unsupported_head_ratio_x_low_visual",
            "unsupported_object_logit",
        ],
    )
    parser.add_argument("--unsupported_component_object_vocab_path", type=str, default="eval_scripts/eval_utils/data/synonyms.txt")
    parser.add_argument("--unsupported_component_prefill_protect_top_k", type=int, default=0)
    parser.add_argument("--unsupported_component_recent_text_window", type=int, default=8)
    parser.add_argument("--unsupported_component_sink_top_k", type=int, default=0)
    parser.add_argument("--unsupported_component_sink_offsets", type=str, default="")
    parser.add_argument("--unsupported_component_all_heads", action="store_true", default=False)
    parser.add_argument("--unsupported_component_query_gate_calibration", type=str, default="")
    parser.add_argument("--unsupported_component_query_gate_top_k", type=int, default=0)
    parser.add_argument("--unsupported_component_query_gate_min_auroc", type=float, default=0.65)
    parser.add_argument(
        "--unsupported_component_query_gate_mode",
        type=str,
        default="off",
        choices=["off", "prefill_hard", "prefill_sigmoid", "decode_hard", "decode_sigmoid"],
    )
    parser.add_argument(
        "--unsupported_component_query_gate_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
    )
    parser.add_argument("--unsupported_component_query_gate_temperature", type=float, default=0.05)
    parser.add_argument("--unsupported_component_query_gate_default", type=float, default=0.0)
    parser.add_argument("--unsupported_component_query_gate_min", type=float, default=0.0)
    parser.add_argument("--unsupported_component_query_gate_power", type=float, default=1.0)
    parser.add_argument("--record_unsupported_component_diagnostics", action="store_true", default=False)
    parser.add_argument("--record_unsupported_component_candidates", action="store_true", default=False)
    parser.add_argument("--record_token_score_diagnostics", action="store_true", default=False)
    parser.add_argument("--unsupported_component_diagnostics_file", type=str, default="")
    parser.add_argument("--unsupported_component_diagnostics_max_records", type=int, default=0)
    parser.add_argument("--head_norm_thresholds_path", type=str, default="")

    args = parser.parse_args()
    if sum([
        args.adaptive_deactivate,
        args.soft_deactivate,
        args.dynamic_deactivate,
        args.attribution_soft_deactivate,
        args.retention_aware_deactivate,
        args.visual_gate_deactivate,
        args.wide_gate_deactivate,
        args.online_value_selector_deactivate,
        args.layer_contrastive_deactivate,
        args.unsupported_component_deactivate,
    ]) > 1:
        raise ValueError("Only one intervention mode can be enabled")
    set_seed(args.seed)
    eval_model(args)
