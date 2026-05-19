# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Literal

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import sys

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _record_query_diagnostics(config, layer_idx, query_states, num_heads):
    if not getattr(config, "record_query_diagnostics", False):
        return
    records = getattr(config, "query_diagnostics", None)
    if records is None:
        return

    layer = int(layer_idx) if layer_idx is not None else -1
    min_layer = getattr(config, "query_record_min_layer", None)
    max_layer = getattr(config, "query_record_max_layer", None)
    if min_layer is not None and layer < int(min_layer):
        return
    if max_layer is not None and layer > int(max_layer):
        return

    if getattr(config, "query_record_all_heads", True):
        diag_heads = range(num_heads)
    else:
        diag_heads = getattr(config, "query_record_heads", None)
        if diag_heads is None:
            return

    batch_idx = int(getattr(config, "query_record_batch_index", 0))
    q_last = query_states[:, :, -1, :].detach().float().cpu()
    if batch_idx < 0 or batch_idx >= q_last.shape[0]:
        return
    q_last = q_last[batch_idx]
    for head in diag_heads:
        head = int(head)
        if head < 0 or head >= num_heads:
            continue
        records.append({
            "layer": layer,
            "head": head,
            "head_key": f"{layer}:{head}",
            "query": q_last[head].clone(),
        })


def _record_head_output_diagnostics(config, layer_idx, head_outputs, num_heads):
    if not getattr(config, "record_head_output_diagnostics", False):
        return
    records = getattr(config, "head_output_diagnostics", None)
    if records is None:
        return
    layer = int(layer_idx) if layer_idx is not None else -1
    min_layer = getattr(config, "head_output_record_min_layer", None)
    max_layer = getattr(config, "head_output_record_max_layer", None)
    if min_layer is not None and layer < int(min_layer):
        return
    if max_layer is not None and layer > int(max_layer):
        return

    if getattr(config, "head_output_record_all_heads", False):
        diag_heads = list(range(num_heads))
    else:
        diag_heads = getattr(config, "head_output_record_heads", None)
        if diag_heads is None:
            return

    batch_idx = int(getattr(config, "head_output_record_batch_index", 0))
    output_last = head_outputs[:, :, -1, :].detach().float().cpu()
    if batch_idx < 0 or batch_idx >= output_last.shape[0]:
        return
    output_last = output_last[batch_idx]
    for head in diag_heads:
        head = int(head)
        if head < 0 or head >= num_heads:
            continue
        records.append({
            "layer": layer,
            "head": head,
            "head_key": f"{layer}:{head}",
            "head_output": output_last[head].clone(),
        })


def _record_residual_diagnostics(config, layer_idx, hidden_states):
    if not getattr(config, "record_residual_diagnostics", False):
        return
    records = getattr(config, "residual_diagnostics", None)
    if records is None:
        return
    layer = int(layer_idx) if layer_idx is not None else -1
    min_layer = getattr(config, "residual_record_min_layer", None)
    max_layer = getattr(config, "residual_record_max_layer", None)
    if min_layer is not None and layer < int(min_layer):
        return
    if max_layer is not None and layer > int(max_layer):
        return

    batch_idx = int(getattr(config, "residual_record_batch_index", 0))
    residual_last = hidden_states[:, -1, :].detach().float().cpu()
    if batch_idx < 0 or batch_idx >= residual_last.shape[0]:
        return
    records.append({
        "layer": layer,
        "head": 0,
        "head_key": f"{layer}:0",
        "residual": residual_last[batch_idx].clone(),
    })


def _apply_query_direction_projection(config, layer_idx, query_states, num_heads):
    if not getattr(config, "query_direction_project", False):
        return query_states
    directions = getattr(config, "query_direction_directions", None)
    if not directions:
        return query_states

    layer = int(layer_idx) if layer_idx is not None else -1
    strength = float(getattr(config, "query_direction_strength", 0.0))
    if abs(strength) <= 0.0:
        return query_states
    strength = min(max(strength, -1.0), 1.0)

    gate_mode = getattr(config, "query_direction_gate_mode", "threshold")
    temperature = max(float(getattr(config, "query_direction_temperature", 0.05)), 1e-6)
    positive_only = bool(getattr(config, "query_direction_positive_only", True))
    thresholds = getattr(config, "query_direction_thresholds", {})
    records = getattr(config, "query_projection_diagnostics", None)
    record_diagnostics = bool(getattr(config, "record_query_projection_diagnostics", False) and records is not None)
    eps = float(getattr(config, "query_direction_eps", 1e-6))

    for head in range(num_heads):
        key = f"{layer}:{head}"
        direction = directions.get(key)
        if direction is None:
            continue
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        direction = direction.to(device=query_states.device, dtype=query_states.dtype)
        direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=eps)

        q = query_states[:, head, -1, :]
        raw_coeff = torch.sum(q * direction, dim=-1, keepdim=True)
        q_norm = q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=eps)
        norm_score = torch.sum(q_norm * direction, dim=-1, keepdim=True)
        threshold = float(thresholds.get(key, 0.0))

        if gate_mode == "none":
            gate = torch.ones_like(raw_coeff)
        elif gate_mode == "positive":
            gate = (raw_coeff > 0).to(query_states.dtype)
        elif gate_mode == "sigmoid":
            gate = torch.sigmoid((norm_score - threshold) / temperature).to(query_states.dtype)
        else:
            gate = (norm_score >= threshold).to(query_states.dtype)

        positive_coeff = (raw_coeff > 0).to(query_states.dtype)
        coeff = raw_coeff
        if positive_only:
            coeff = torch.clamp(coeff, min=0.0)
        active_projection = (torch.abs(gate * coeff) > eps).to(query_states.dtype)
        projected_q = q - strength * gate * coeff * direction
        if record_diagnostics:
            projected_raw_coeff = torch.sum(projected_q * direction, dim=-1, keepdim=True)
            projected_q_norm = projected_q / torch.clamp(torch.linalg.vector_norm(projected_q, dim=-1, keepdim=True), min=eps)
            projected_norm_score = torch.sum(projected_q_norm * direction, dim=-1, keepdim=True)
            q_delta = projected_q - q
            q_delta_norm = torch.linalg.vector_norm(q_delta, dim=-1, keepdim=True)
            q_norm_value = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
            records.append({
                "kind": "query_projection",
                "layer": layer,
                "head": head,
                "head_key": key,
                "strength": strength,
                "gate_mode": gate_mode,
                "gate": gate.detach().float().cpu().item(),
                "positive_only": float(positive_only),
                "positive_coeff": positive_coeff.detach().float().cpu().item(),
                "effective_coeff": coeff.detach().float().cpu().item(),
                "active_projection": active_projection.detach().float().cpu().item(),
                "threshold": threshold,
                "raw_score_before": raw_coeff.detach().float().cpu().item(),
                "raw_score_after": projected_raw_coeff.detach().float().cpu().item(),
                "raw_score_delta": (raw_coeff - projected_raw_coeff).detach().float().cpu().item(),
                "normalized_score_before": norm_score.detach().float().cpu().item(),
                "normalized_score_after": projected_norm_score.detach().float().cpu().item(),
                "normalized_score_delta": (norm_score - projected_norm_score).detach().float().cpu().item(),
                "q_delta_norm": q_delta_norm.detach().float().cpu().item(),
                "q_norm": q_norm_value.detach().float().cpu().item(),
                "relative_q_delta": (q_delta_norm / torch.clamp(q_norm_value, min=eps)).detach().float().cpu().item(),
            })
        query_states[:, head, -1, :] = projected_q
    return query_states


def _apply_head_output_direction_projection(config, layer_idx, head_outputs, num_heads):
    if not getattr(config, "head_output_direction_project", False):
        return head_outputs
    directions = getattr(config, "head_output_direction_directions", None)
    if not directions:
        return head_outputs

    layer = int(layer_idx) if layer_idx is not None else -1
    strength = float(getattr(config, "head_output_direction_strength", 0.0))
    if abs(strength) <= 0.0:
        return head_outputs
    strength = min(max(strength, -1.0), 1.0)

    gate_mode = getattr(config, "head_output_direction_gate_mode", "threshold")
    temperature = max(float(getattr(config, "head_output_direction_temperature", 0.05)), 1e-6)
    positive_only = bool(getattr(config, "head_output_direction_positive_only", True))
    thresholds = getattr(config, "head_output_direction_thresholds", {})
    records = getattr(config, "head_output_projection_diagnostics", None)
    record_diagnostics = bool(getattr(config, "record_head_output_projection_diagnostics", False) and records is not None)
    eps = float(getattr(config, "head_output_direction_eps", 1e-6))

    for head in range(num_heads):
        key = f"{layer}:{head}"
        direction = directions.get(key)
        if direction is None:
            continue
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        direction = direction.to(device=head_outputs.device, dtype=head_outputs.dtype)
        direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=eps)

        output = head_outputs[:, head, -1, :]
        raw_coeff = torch.sum(output * direction, dim=-1, keepdim=True)
        output_normed = output / torch.clamp(torch.linalg.vector_norm(output, dim=-1, keepdim=True), min=eps)
        norm_score = torch.sum(output_normed * direction, dim=-1, keepdim=True)
        threshold = float(thresholds.get(key, 0.0))

        if gate_mode == "none":
            gate = torch.ones_like(raw_coeff)
        elif gate_mode == "positive":
            gate = (raw_coeff > 0).to(head_outputs.dtype)
        elif gate_mode == "sigmoid":
            gate = torch.sigmoid((norm_score - threshold) / temperature).to(head_outputs.dtype)
        else:
            gate = (norm_score >= threshold).to(head_outputs.dtype)

        positive_coeff = (raw_coeff > 0).to(head_outputs.dtype)
        coeff = raw_coeff
        if positive_only:
            coeff = torch.clamp(coeff, min=0.0)
        active_projection = (torch.abs(gate * coeff) > eps).to(head_outputs.dtype)
        projected_output = output - strength * gate * coeff * direction
        if record_diagnostics:
            projected_raw_coeff = torch.sum(projected_output * direction, dim=-1, keepdim=True)
            projected_output_normed = projected_output / torch.clamp(
                torch.linalg.vector_norm(projected_output, dim=-1, keepdim=True),
                min=eps,
            )
            projected_norm_score = torch.sum(projected_output_normed * direction, dim=-1, keepdim=True)
            output_delta = projected_output - output
            output_delta_norm = torch.linalg.vector_norm(output_delta, dim=-1, keepdim=True)
            output_norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
            records.append({
                "kind": "head_output_projection",
                "layer": layer,
                "head": head,
                "head_key": key,
                "strength": strength,
                "gate_mode": gate_mode,
                "gate": gate.detach().float().cpu().item(),
                "positive_only": float(positive_only),
                "positive_coeff": positive_coeff.detach().float().cpu().item(),
                "effective_coeff": coeff.detach().float().cpu().item(),
                "active_projection": active_projection.detach().float().cpu().item(),
                "threshold": threshold,
                "raw_score_before": raw_coeff.detach().float().cpu().item(),
                "raw_score_after": projected_raw_coeff.detach().float().cpu().item(),
                "raw_score_delta": (raw_coeff - projected_raw_coeff).detach().float().cpu().item(),
                "normalized_score_before": norm_score.detach().float().cpu().item(),
                "normalized_score_after": projected_norm_score.detach().float().cpu().item(),
                "normalized_score_delta": (norm_score - projected_norm_score).detach().float().cpu().item(),
                "head_output_delta_norm": output_delta_norm.detach().float().cpu().item(),
                "head_output_norm": output_norm.detach().float().cpu().item(),
                "relative_head_output_delta": (
                    output_delta_norm / torch.clamp(output_norm, min=eps)
                ).detach().float().cpu().item(),
            })
        head_outputs[:, head, -1, :] = projected_output
    return head_outputs


def _apply_residual_direction_projection(config, layer_idx, hidden_states):
    if not getattr(config, "residual_direction_project", False):
        return hidden_states
    directions = getattr(config, "residual_direction_directions", None)
    if not directions:
        return hidden_states

    layer = int(layer_idx) if layer_idx is not None else -1
    key = f"{layer}:0"
    direction = directions.get(key)
    if direction is None:
        return hidden_states

    strength = float(getattr(config, "residual_direction_strength", 0.0))
    if abs(strength) <= 0.0:
        return hidden_states
    strength = min(max(strength, -1.0), 1.0)

    gate_mode = getattr(config, "residual_direction_gate_mode", "threshold")
    temperature = max(float(getattr(config, "residual_direction_temperature", 0.05)), 1e-6)
    positive_only = bool(getattr(config, "residual_direction_positive_only", True))
    thresholds = getattr(config, "residual_direction_thresholds", {})
    records = getattr(config, "residual_projection_diagnostics", None)
    record_diagnostics = bool(getattr(config, "record_residual_projection_diagnostics", False) and records is not None)
    eps = float(getattr(config, "residual_direction_eps", 1e-6))

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor(direction)
    direction = direction.to(device=hidden_states.device, dtype=hidden_states.dtype)
    direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=eps)

    residual = hidden_states[:, -1, :]
    raw_coeff = torch.sum(residual * direction, dim=-1, keepdim=True)
    residual_normed = residual / torch.clamp(torch.linalg.vector_norm(residual, dim=-1, keepdim=True), min=eps)
    norm_score = torch.sum(residual_normed * direction, dim=-1, keepdim=True)
    threshold = float(thresholds.get(key, 0.0))

    if gate_mode == "none":
        gate = torch.ones_like(raw_coeff)
    elif gate_mode == "positive":
        gate = (raw_coeff > 0).to(hidden_states.dtype)
    elif gate_mode == "sigmoid":
        gate = torch.sigmoid((norm_score - threshold) / temperature).to(hidden_states.dtype)
    else:
        gate = (norm_score >= threshold).to(hidden_states.dtype)

    positive_coeff = (raw_coeff > 0).to(hidden_states.dtype)
    coeff = raw_coeff
    if positive_only:
        coeff = torch.clamp(coeff, min=0.0)
    active_projection = (torch.abs(gate * coeff) > eps).to(hidden_states.dtype)
    projected_residual = residual - strength * gate * coeff * direction
    if record_diagnostics:
        projected_raw_coeff = torch.sum(projected_residual * direction, dim=-1, keepdim=True)
        projected_residual_normed = projected_residual / torch.clamp(
            torch.linalg.vector_norm(projected_residual, dim=-1, keepdim=True),
            min=eps,
        )
        projected_norm_score = torch.sum(projected_residual_normed * direction, dim=-1, keepdim=True)
        residual_delta = projected_residual - residual
        residual_delta_norm = torch.linalg.vector_norm(residual_delta, dim=-1, keepdim=True)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        records.append({
            "kind": "residual_projection",
            "layer": layer,
            "head": 0,
            "head_key": key,
            "strength": strength,
            "gate_mode": gate_mode,
            "gate": gate.detach().float().cpu().item(),
            "positive_only": float(positive_only),
            "positive_coeff": positive_coeff.detach().float().cpu().item(),
            "effective_coeff": coeff.detach().float().cpu().item(),
            "active_projection": active_projection.detach().float().cpu().item(),
            "threshold": threshold,
            "raw_score_before": raw_coeff.detach().float().cpu().item(),
            "raw_score_after": projected_raw_coeff.detach().float().cpu().item(),
            "raw_score_delta": (raw_coeff - projected_raw_coeff).detach().float().cpu().item(),
            "normalized_score_before": norm_score.detach().float().cpu().item(),
            "normalized_score_after": projected_norm_score.detach().float().cpu().item(),
            "normalized_score_delta": (norm_score - projected_norm_score).detach().float().cpu().item(),
            "residual_delta_norm": residual_delta_norm.detach().float().cpu().item(),
            "residual_norm": residual_norm.detach().float().cpu().item(),
            "relative_residual_delta": (
                residual_delta_norm / torch.clamp(residual_norm, min=eps)
            ).detach().float().cpu().item(),
        })
    hidden_states[:, -1, :] = projected_residual
    return hidden_states


def _record_query_attention_projection_diagnostics(
    config,
    layer_idx,
    original_query_states,
    projected_query_states,
    key_states,
    attention_mask,
    num_heads,
    head_dim,
):
    if original_query_states is None:
        return
    records = getattr(config, "query_projection_diagnostics", None)
    if not getattr(config, "record_query_projection_diagnostics", False) or records is None:
        return
    directions = getattr(config, "query_direction_directions", None)
    if not directions:
        return

    layer = int(layer_idx) if layer_idx is not None else -1
    eps = float(getattr(config, "query_direction_eps", 1e-6))
    if key_states.shape[1] != num_heads:
        num_key_value_groups = max(num_heads // key_states.shape[1], 1)
        key_states = repeat_kv(key_states, num_key_value_groups)

    for head in range(num_heads):
        key = f"{layer}:{head}"
        if key not in directions:
            continue
        q_original = original_query_states[:, head, -1:, :]
        q_projected = projected_query_states[:, head, -1:, :]
        head_keys = key_states[:, head, :, :]
        original_logits = torch.matmul(q_original, head_keys.transpose(1, 2)) / math.sqrt(head_dim)
        projected_logits = torch.matmul(q_projected, head_keys.transpose(1, 2)) / math.sqrt(head_dim)
        if (
            attention_mask is not None
            and attention_mask.dim() == 4
            and attention_mask.shape[-1] == original_logits.shape[-1]
        ):
            original_logits = original_logits + attention_mask[:, :, -1, :]
            projected_logits = projected_logits + attention_mask[:, :, -1, :]

        original_attention = nn.functional.softmax(original_logits.float(), dim=-1)
        projected_attention = nn.functional.softmax(projected_logits.float(), dim=-1)
        attention_kl = torch.sum(
            original_attention * (
                torch.log(original_attention + eps) - torch.log(projected_attention + eps)
            ),
            dim=-1,
        )
        attention_l1 = torch.sum(torch.abs(original_attention - projected_attention), dim=-1)
        logit_delta = projected_logits.float() - original_logits.float()
        logit_delta_norm = torch.linalg.vector_norm(logit_delta, dim=-1)
        original_logit_norm = torch.linalg.vector_norm(original_logits.float(), dim=-1)
        records.append({
            "kind": "attention_projection",
            "layer": layer,
            "head": head,
            "head_key": key,
            "attention_logit_delta_norm": logit_delta_norm.detach().float().cpu().item(),
            "relative_attention_logit_delta": (
                logit_delta_norm / torch.clamp(original_logit_norm, min=eps)
            ).detach().float().cpu().item(),
            "attention_kl": attention_kl.detach().float().cpu().item(),
            "attention_l1": attention_l1.detach().float().cpu().item(),
        })


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )

# TODO: newly added function
def create_causal_attention_mask(attn_mask, num_heads):
    """
    Create a causal attention mask compatible with flash attention.
    
    Args:
        attn_mask: Attention mask tensor of shape [batch_size, sequence_length]
        num_heads: Number of attention heads
    
    Returns:
        combined_mask: Causal attention mask of shape [batch_size, num_heads, sequence_length, sequence_length]
                      where each position i can only attend to positions 0 to i
    """
    import pdb; pdb.set_trace()
    batch_size, seq_len = attn_mask.shape
    
    # Expand attention mask to match shape [batch_size, num_heads, seq_len, seq_len]
    expanded_attn_mask = attn_mask[:, None, None, :].expand(batch_size, num_heads, seq_len, seq_len)
    
    # Create causal mask where each position i can only attend to positions 0 to i
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(attn_mask.device)
    
    # Combine attention mask with causal mask to enforce both padding and causality constraints
    combined_mask = expanded_attn_mask & causal_mask
    
    return combined_mask

# TODO: newly added function
def calculate_attention_weights(query_states, key_states, attention_mask, num_heads, head_dim):
    """
    Calculate attention weights between query and key states with causal masking.
    
    Args:
        query_states: Query tensor of shape [batch_size, sequence_length, num_heads, head_dim]
        key_states: Key tensor of shape [batch_size, sequence_length, num_heads, head_dim] 
        attention_mask: Attention mask tensor of shape [batch_size, sequence_length]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
    
    Returns:
        attention_weights: Tensor of shape [batch_size, num_heads, sequence_length, sequence_length]
    """
    # Reshape tensors to [batch_size, num_heads, sequence_length, head_dim]
    query_states = query_states.permute(0, 2, 1, 3)
    key_states = key_states.permute(0, 2, 1, 3)

    # Calculate raw attention scores
    attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
    attention_scores = attention_scores / math.sqrt(head_dim)  # Apply scaling factor

    # Create and apply causal mask
    causal_mask = create_causal_attention_mask(attention_mask, num_heads)
    attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))

    # Apply softmax to get attention weights
    attention_weights = nn.functional.softmax(attention_scores, dim=-1)

    del query_states, key_states, attention_scores, causal_mask

    return attention_weights

# TODO: newly added function
def calculate_attention_statistics(
    attention_weights: torch.Tensor,
    labels: torch.Tensor,
    head_list: list[int],
    model_type: Literal["llava", "minigpt4"] = "llava",
    loss: Literal["maximize_entropy", "maximize_img", "minimize_txt"] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate attention entropy and related statistics for vision-language models.
    
    Args:
        attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
        labels: Labels tensor of shape (batch_size, seq_len)
        head_list: List of attention head indices to analyze
        model_type: Type of vision-language model ("llava" or "minigpt4")
        loss: Type of loss to compute (None or one of "maximize_entropy", "maximize_img", "minimize_txt")
    
    Returns:
        Tuple containing:
        - attention loss (if loss type specified, else entropy)
        - mean image attention score
        - mean text attention score  
        - mean entropy
    """
    # attention_weights = calculate_attention_weights(query_states, key_states, attention_mask, num_heads, head_dim)

    # Get dimensions
    batch_size, seq_len = labels.size()
    
    # Select attention weights for specified heads
    attn_weights = attention_weights[:, head_list]

    # Create mask for valid label positions and expand to match attention dimensions
    label_mask = (labels != -100).unsqueeze(1).expand(batch_size, len(head_list), seq_len)
    
    # Filter attention weights to only valid positions
    attn_weights = attn_weights[label_mask.nonzero(as_tuple=True)]

    # Create image and text position masks based on model type
    img_mask = torch.zeros(attn_weights.size(0), seq_len, device=labels.device)
    if model_type == "llava":
        img_start, img_len = 35, 576
    elif model_type == "minigpt4":
        img_start, img_len = 7, 64
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    img_mask[:, img_start:img_start + img_len] = 1.0
    txt_mask = 1.0 - img_mask

    # Calculate attention scores for image and text
    img_attn_score = (attn_weights * img_mask).sum(dim=-1)
    txt_attn_score = (attn_weights * txt_mask).sum(dim=-1)

    # Calculate entropy
    entropy = -torch.mean(
        img_attn_score * torch.log(img_attn_score) +
        txt_attn_score * torch.log(txt_attn_score)
    )

    # Calculate loss based on specified type
    if loss == "maximize_entropy":
        attn_loss = -entropy
    elif loss == "maximize_img":
        attn_loss = -torch.log(img_attn_score + 1e-8).mean()
    elif loss == "minimize_txt":
        attn_loss = torch.log(txt_attn_score + 1e-8).mean()
    elif loss is None:
        attn_loss = torch.log(txt_attn_score + 1e-8).mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss}")
    
    del attn_weights

    return (
        attn_loss,
        img_attn_score.mean(),
        txt_attn_score.mean(),
        entropy.mean()
    )

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size # 14336
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        output_attention_statistics: bool = False,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
        head_list: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        original_query_states = query_states.clone() if getattr(self.config, "record_query_projection_diagnostics", False) else None
        _record_query_diagnostics(self.config, self.layer_idx, query_states, self.num_heads)
        query_states = _apply_query_direction_projection(self.config, self.layer_idx, query_states, self.num_heads)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if original_query_states is not None:
            original_query_states, _ = apply_rotary_pos_emb(original_query_states, key_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        _record_query_attention_projection_diagnostics(
            self.config,
            self.layer_idx,
            original_query_states,
            query_states,
            key_states,
            attention_mask,
            self.num_heads,
            self.head_dim,
        )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # TODO: adaptive deactivate of hallucination heads
        if getattr(self.config, "adaptive_deactivate", False):
            if head_list is not None:
                for head in head_list:
                    aggre_attention = torch.sum(attn_weights[:, head, -1, self.config.img_start_pos+self.config.img_length:])
                    if aggre_attention >= self.config.adhh_threshold:
                        attn_weights[:, head, -1, self.config.img_start_pos+self.config.img_length:] = 0

        # Soft routing variant of AD-HH: keep the same fixed heads and threshold,
        # but continuously downscale text attention instead of zeroing it.
        if getattr(self.config, "soft_deactivate", False):
            if head_list is not None:
                text_start_idx = self.config.img_start_pos + self.config.img_length
                temperature = max(float(getattr(self.config, "soft_temperature", 0.05)), 1e-6)
                gamma = float(getattr(self.config, "soft_gamma", 0.5))
                threshold = float(getattr(self.config, "adhh_threshold", 0.0))
                for head in head_list:
                    text_attention = attn_weights[:, head, -1, text_start_idx:]
                    text_mass = torch.sum(text_attention, dim=-1, keepdim=True)
                    alpha = torch.sigmoid((text_mass - threshold) / temperature).to(attn_weights.dtype)
                    text_attention *= (1.0 - gamma * alpha)

        if getattr(self.config, "fixed_strength_deactivate", False):
            if head_list is not None:
                text_start_idx = self.config.img_start_pos + self.config.img_length
                threshold = float(getattr(self.config, "adhh_threshold", 0.0))
                strength = float(getattr(self.config, "fixed_suppression_strength", 1.0))
                strength = min(max(strength, 0.0), 1.0)
                for head in head_list:
                    text_attention = attn_weights[:, head, -1, text_start_idx:]
                    text_mass = torch.sum(text_attention, dim=-1, keepdim=True)
                    trigger = (text_mass >= threshold).to(attn_weights.dtype)
                    text_attention *= (1.0 - strength * trigger)

        # Training-free dynamic soft routing. This keeps AD-HH's fixed head set,
        # but makes suppression strength vary by head and decoding state.
        if getattr(self.config, "dynamic_deactivate", False):
            if head_list is not None:
                img_slice = slice(self.config.img_start_pos, self.config.img_start_pos + self.config.img_length)
                text_start_idx = self.config.img_start_pos + self.config.img_length
                temperature = max(float(getattr(self.config, "dynamic_temperature", 0.05)), 1e-6)
                gamma = float(getattr(self.config, "dynamic_gamma", 1.0))
                threshold = float(getattr(self.config, "adhh_threshold", 0.0))
                eps = float(getattr(self.config, "dynamic_eps", 1e-6))
                margins = []
                ratios = []
                text_masses = []
                img_masses = []
                for head in head_list:
                    text_mass = torch.sum(attn_weights[:, head, -1, text_start_idx:], dim=-1, keepdim=True)
                    img_mass = torch.sum(attn_weights[:, head, -1, img_slice], dim=-1, keepdim=True)
                    margin = (text_mass - threshold) / temperature
                    ratio = torch.log((text_mass + eps) / (img_mass + eps))
                    margins.append(margin)
                    ratios.append(ratio)
                    text_masses.append(text_mass)
                    img_masses.append(img_mass)

                trigger_mask = [text_mass >= threshold for text_mass in text_masses]
                consensus = torch.stack([mask.to(attn_weights.dtype) for mask in trigger_mask], dim=0).mean(dim=0)
                margin_weight = float(getattr(self.config, "dynamic_margin_weight", 1.0))
                ratio_weight = float(getattr(self.config, "dynamic_ratio_weight", 0.25))
                consensus_weight = float(getattr(self.config, "dynamic_consensus_weight", 0.5))
                bias = float(getattr(self.config, "dynamic_bias", 0.0))

                for idx, head in enumerate(head_list):
                    risk = margin_weight * margins[idx] + ratio_weight * ratios[idx] + consensus_weight * consensus + bias
                    strength = gamma * torch.sigmoid(risk).to(attn_weights.dtype)
                    attn_weights[:, head, -1, text_start_idx:] *= (1.0 - strength)

        if getattr(self.config, "retention_aware_deactivate", False):
            if head_list is not None:
                text_start_idx = self.config.img_start_pos + self.config.img_length
                priors = getattr(self.config, "head_attribution_priors", {})
                threshold = float(getattr(self.config, "adhh_threshold", 0.0))
                temperature = max(float(getattr(self.config, "retention_soft_temperature", 0.05)), 1e-6)
                soft_gamma = float(getattr(self.config, "retention_soft_gamma", 0.75))
                retention_rho = float(getattr(self.config, "retention_rho", 0.1))
                retention_lambda = float(getattr(self.config, "retention_lambda", 1.0))
                retention_feature = getattr(self.config, "retention_feature", "mean_prior_text_mass")
                retention_policy_mode = getattr(self.config, "retention_policy_mode", "hard_or_soft")
                eps = float(getattr(self.config, "retention_eps", 1e-6))

                head_values = []
                trigger_values = []
                prior_text_values = []
                excess_values = []
                weighted_excess_values = []
                weighted_trigger_values = []
                for head in head_list:
                    key = f"{int(self.layer_idx)}:{int(head)}"
                    prior = float(priors.get(key, 1.0))
                    text_attention = attn_weights[:, head, -1, text_start_idx:]
                    text_mass = torch.sum(text_attention, dim=-1, keepdim=True)
                    trigger = (text_mass >= threshold).to(attn_weights.dtype)
                    excess = torch.clamp(text_mass - threshold, min=0.0)
                    alpha = torch.sigmoid((text_mass - threshold) / temperature).to(attn_weights.dtype)
                    head_values.append((head, text_attention, prior, trigger, excess, alpha))
                    trigger_values.append(trigger)
                    prior_text_values.append(text_mass * prior)
                    excess_values.append(excess)
                    weighted_excess_values.append(excess * prior)
                    weighted_trigger_values.append(trigger * prior)

                if retention_feature == "trigger_frac":
                    retention_risk = torch.stack(trigger_values, dim=0).mean(dim=0)
                elif retention_feature == "weighted_trigger_count":
                    retention_risk = torch.stack(weighted_trigger_values, dim=0).sum(dim=0)
                elif retention_feature == "weighted_excess":
                    retention_risk = torch.stack(weighted_excess_values, dim=0).sum(dim=0)
                elif retention_feature == "mean_excess":
                    retention_risk = torch.stack(excess_values, dim=0).mean(dim=0)
                else:
                    retention_risk = torch.stack(prior_text_values, dim=0).mean(dim=0)

                high_retention = (retention_risk >= retention_rho).to(attn_weights.dtype)
                for head, text_attention, prior, trigger, excess, alpha in head_values:
                    if retention_policy_mode == "cap":
                        strength = soft_gamma * prior * alpha / (1.0 + retention_lambda * retention_risk + eps)
                    else:
                        hard_strength = trigger
                        soft_strength = soft_gamma * alpha
                        strength = high_retention * soft_strength + (1.0 - high_retention) * hard_strength
                    strength = torch.clamp(strength, min=0.0, max=1.0).to(attn_weights.dtype)
                    text_attention *= (1.0 - strength)

        if getattr(self.config, "visual_gate_deactivate", False):
            if head_list is not None:
                img_slice = slice(self.config.img_start_pos, self.config.img_start_pos + self.config.img_length)
                text_start_idx = self.config.img_start_pos + self.config.img_length
                priors = getattr(self.config, "head_attribution_priors", {})
                thresholds = getattr(self.config, "head_text_thresholds", {})
                gamma = float(getattr(self.config, "visual_gate_gamma", 1.0))
                beta = float(getattr(self.config, "visual_gate_beta", 0.75))
                v0 = float(getattr(self.config, "visual_gate_v0", 0.5))
                temperature = max(float(getattr(self.config, "visual_gate_temperature", 0.15)), 1e-6)
                proxy = getattr(self.config, "visual_gate_proxy", "value")
                recent_weight = float(getattr(self.config, "visual_gate_recent_weight", 0.0))
                recent_window = int(getattr(self.config, "visual_gate_recent_window", 16))
                default_low = float(getattr(self.config, "visual_gate_tau_low", getattr(self.config, "adhh_threshold", 0.4)))
                default_high = float(getattr(self.config, "visual_gate_tau_high", 0.9))
                eps = float(getattr(self.config, "visual_gate_eps", 1e-6))
                recent_start = min(max(kv_seq_len - recent_window, text_start_idx), kv_seq_len)
                recent_slice = slice(recent_start, kv_seq_len)

                for head in head_list:
                    key = f"{int(self.layer_idx)}:{int(head)}"
                    prior = float(priors.get(key, 1.0))
                    threshold = thresholds.get(key, {})
                    low = float(threshold.get("low", default_low))
                    high = float(threshold.get("high", default_high))
                    if high <= low:
                        high = low + eps

                    head_weights = attn_weights[:, head, -1, :]
                    head_values = value_states[:, head, :, :]
                    text_attention = head_weights[:, text_start_idx:]
                    img_attention = head_weights[:, img_slice]
                    recent_attention = head_weights[:, recent_slice]

                    text_mass = torch.sum(text_attention, dim=-1, keepdim=True)
                    img_mass = torch.sum(img_attention, dim=-1, keepdim=True)
                    recent_mass = torch.sum(recent_attention, dim=-1, keepdim=True)
                    excess = torch.clamp((text_mass - low) / (high - low), min=0.0, max=1.0)
                    visual_mass_ratio = img_mass / (img_mass + text_mass + eps)

                    if proxy in {"value", "value_recent"}:
                        text_value = torch.bmm(text_attention.float().unsqueeze(1), head_values[:, text_start_idx:, :].float()).squeeze(1)
                        img_value = torch.bmm(img_attention.float().unsqueeze(1), head_values[:, img_slice, :].float()).squeeze(1)
                        text_value_norm = torch.linalg.vector_norm(text_value, dim=-1, keepdim=True)
                        img_value_norm = torch.linalg.vector_norm(img_value, dim=-1, keepdim=True)
                        visual_proxy = img_value_norm / (img_value_norm + text_value_norm + eps)
                    else:
                        visual_proxy = visual_mass_ratio

                    if proxy == "value_recent":
                        recent_ratio = recent_mass / (text_mass + eps)
                        visual_proxy = torch.clamp(visual_proxy + recent_weight * recent_ratio, min=0.0, max=1.0)

                    retention_gate = torch.sigmoid((visual_proxy - v0) / temperature).to(attn_weights.dtype)
                    strength = gamma * prior * excess * (1.0 - beta * retention_gate)
                    strength = torch.clamp(strength, min=0.0, max=1.0).to(attn_weights.dtype)
                    text_attention *= (1.0 - strength)

        if getattr(self.config, "wide_gate_deactivate", False):
            if head_list is not None:
                text_start_idx = self.config.img_start_pos + self.config.img_length
                mode = getattr(self.config, "wide_gate_mode", "hard")
                gate_feature = getattr(self.config, "wide_gate_feature", "text_norm")
                text_tau = float(getattr(self.config, "wide_gate_text_tau", getattr(self.config, "adhh_threshold", 0.4)))
                text_high = float(getattr(self.config, "wide_gate_text_high", 0.9))
                gamma = float(getattr(self.config, "wide_gate_gamma", 1.0))
                default_norm_threshold = float(getattr(self.config, "wide_gate_norm_threshold", 0.0))
                default_norm_low = float(getattr(self.config, "wide_gate_norm_low", default_norm_threshold))
                default_norm_high = float(getattr(self.config, "wide_gate_norm_high", max(default_norm_threshold + 1e-6, 1.0)))
                norm_source = getattr(self.config, "wide_gate_norm_source", "text_value")
                norm_thresholds = getattr(self.config, "head_norm_thresholds", {})
                eps = float(getattr(self.config, "wide_gate_eps", 1e-6))
                text_den = max(text_high - text_tau, eps)

                for head in head_list:
                    key = f"{int(self.layer_idx)}:{int(head)}"
                    head_weights = attn_weights[:, head, -1, :]
                    head_values = value_states[:, head, :, :]
                    text_attention = head_weights[:, text_start_idx:]
                    text_mass = torch.sum(text_attention, dim=-1, keepdim=True)

                    text_value = torch.bmm(
                        text_attention.float().unsqueeze(1),
                        head_values[:, text_start_idx:, :].float(),
                    ).squeeze(1)
                    if norm_source == "head_output":
                        norm_value = torch.bmm(
                            head_weights.float().unsqueeze(1),
                            head_values.float(),
                        ).squeeze(1)
                    else:
                        norm_value = text_value
                    norm = torch.linalg.vector_norm(norm_value, dim=-1, keepdim=True)

                    threshold_data = norm_thresholds.get(key, {})
                    norm_threshold = float(threshold_data.get("threshold", default_norm_threshold))
                    norm_low = float(threshold_data.get("low", default_norm_low))
                    norm_high = float(threshold_data.get("high", default_norm_high))
                    if norm_high <= norm_low:
                        norm_high = norm_low + eps

                    text_gate = (text_mass >= text_tau).to(attn_weights.dtype)
                    norm_gate = (norm >= norm_threshold).to(attn_weights.dtype)
                    if gate_feature == "text":
                        gate = text_gate
                    elif gate_feature == "norm":
                        gate = norm_gate
                    else:
                        gate = text_gate * norm_gate

                    if mode == "continuous":
                        text_excess = torch.clamp((text_mass - text_tau) / text_den, min=0.0, max=1.0)
                        norm_excess = torch.clamp((norm - norm_low) / (norm_high - norm_low), min=0.0, max=1.0)
                        if gate_feature == "text":
                            strength = gamma * text_excess
                        elif gate_feature == "norm":
                            strength = gamma * norm_excess
                        else:
                            strength = gamma * text_excess * norm_excess
                    else:
                        strength = gate
                    strength = torch.clamp(strength, min=0.0, max=1.0).to(attn_weights.dtype)
                    text_attention *= (1.0 - strength)

        if getattr(self.config, "online_value_selector_deactivate", False):
            if head_list is not None:
                text_start_idx = self.config.img_start_pos + self.config.img_length
                mode = getattr(self.config, "online_value_selector_mode", "continuous")
                text_tau = float(getattr(self.config, "online_value_selector_text_tau", getattr(self.config, "adhh_threshold", 0.4)))
                gamma = float(getattr(self.config, "online_value_selector_gamma", 1.0))
                layer_top_k = int(getattr(self.config, "online_value_selector_layer_top_k", 1))
                require_text_trigger = bool(getattr(self.config, "online_value_selector_require_text_trigger", True))
                soft_threshold = float(getattr(self.config, "online_value_selector_soft_threshold", 0.25))
                hard_threshold = float(getattr(self.config, "online_value_selector_hard_threshold", 0.75))
                default_norm_threshold = float(getattr(self.config, "online_value_selector_norm_threshold", 0.0))
                default_norm_low = float(getattr(self.config, "online_value_selector_norm_low", default_norm_threshold))
                default_norm_high = float(getattr(self.config, "online_value_selector_norm_high", max(default_norm_threshold + 1e-6, 1.0)))
                norm_source = getattr(self.config, "online_value_selector_norm_source", "text_value")
                norm_thresholds = getattr(self.config, "head_norm_thresholds", {})
                eps = float(getattr(self.config, "online_value_selector_eps", 1e-6))

                candidates = []
                for head in head_list:
                    key = f"{int(self.layer_idx)}:{int(head)}"
                    head_weights = attn_weights[:, head, -1, :]
                    head_values = value_states[:, head, :, :]
                    text_attention = head_weights[:, text_start_idx:]
                    text_mass = torch.sum(text_attention, dim=-1, keepdim=True)

                    text_value = torch.bmm(
                        text_attention.float().unsqueeze(1),
                        head_values[:, text_start_idx:, :].float(),
                    ).squeeze(1)
                    if norm_source == "head_output":
                        norm_value = torch.bmm(
                            head_weights.float().unsqueeze(1),
                            head_values.float(),
                        ).squeeze(1)
                    else:
                        norm_value = text_value
                    norm = torch.linalg.vector_norm(norm_value, dim=-1, keepdim=True)

                    threshold_data = norm_thresholds.get(key, {})
                    norm_threshold = float(threshold_data.get("threshold", default_norm_threshold))
                    norm_low = float(threshold_data.get("low", default_norm_low))
                    norm_high = float(threshold_data.get("high", default_norm_high))
                    if norm_high <= norm_low:
                        norm_high = norm_low + eps

                    text_gate = (text_mass >= text_tau).to(attn_weights.dtype)
                    norm_gate = (norm >= norm_threshold).to(attn_weights.dtype)
                    norm_excess = torch.clamp((norm - norm_low) / (norm_high - norm_low), min=0.0, max=1.0)
                    rank_score = norm
                    if require_text_trigger:
                        rank_score = torch.where(
                            text_gate.bool(),
                            rank_score,
                            torch.full_like(rank_score, float("-inf")),
                        )
                    candidates.append({
                        "head": head,
                        "text_attention": text_attention,
                        "text_gate": text_gate,
                        "norm_gate": norm_gate,
                        "norm_excess": norm_excess,
                        "rank_score": rank_score,
                    })

                if layer_top_k > 0 and len(candidates) > layer_top_k:
                    # Caption eval uses batch size 1. Keeping selection in Python avoids
                    # an extra forward while still making the action depend on online state.
                    scored = []
                    for idx, item in enumerate(candidates):
                        score = item["rank_score"].detach().float().cpu().item()
                        if math.isfinite(score):
                            scored.append((score, idx))
                    selected_indices = {idx for _, idx in sorted(scored, reverse=True)[:layer_top_k]}
                else:
                    selected_indices = set()
                    for idx, item in enumerate(candidates):
                        score = item["rank_score"].detach().float().cpu().item()
                        if math.isfinite(score):
                            selected_indices.add(idx)

                for idx, item in enumerate(candidates):
                    if idx not in selected_indices:
                        continue
                    if mode == "hard":
                        strength = item["text_gate"] if require_text_trigger else item["norm_gate"]
                    elif mode == "hybrid":
                        hard_mask = (item["norm_excess"] >= hard_threshold).to(attn_weights.dtype)
                        soft_mask = (item["norm_excess"] >= soft_threshold).to(attn_weights.dtype)
                        soft_strength = gamma * item["norm_excess"] * soft_mask
                        strength = hard_mask + (1.0 - hard_mask) * soft_strength
                        if require_text_trigger:
                            strength = strength * item["text_gate"]
                    else:
                        strength = gamma * item["norm_excess"]
                        if require_text_trigger:
                            strength = strength * item["text_gate"]
                    strength = torch.clamp(strength, min=0.0, max=1.0).to(attn_weights.dtype)
                    item["text_attention"] *= (1.0 - strength)

        if getattr(self.config, "attribution_soft_deactivate", False):
            if head_list is not None:
                text_start_idx = self.config.img_start_pos + self.config.img_length
                priors = getattr(self.config, "head_attribution_priors", {})
                thresholds = getattr(self.config, "head_text_thresholds", {})
                gamma = float(getattr(self.config, "attribution_soft_gamma", 1.0))
                mode = getattr(self.config, "attribution_soft_mode", "linear")
                default_low = float(getattr(self.config, "attribution_tau_low", getattr(self.config, "adhh_threshold", 0.4)))
                default_high = float(getattr(self.config, "attribution_tau_high", 0.9))
                eps = float(getattr(self.config, "attribution_soft_eps", 1e-6))
                head_values = []
                for head in head_list:
                    key = f"{int(self.layer_idx)}:{int(head)}"
                    prior = float(priors.get(key, 1.0))
                    threshold = thresholds.get(key, {})
                    low = float(threshold.get("low", default_low))
                    high = float(threshold.get("high", default_high))
                    if high <= low:
                        high = low + eps
                    text_attention = attn_weights[:, head, -1, text_start_idx:]
                    text_mass = torch.sum(text_attention, dim=-1, keepdim=True)
                    excess = torch.clamp((text_mass - low) / (high - low), min=0.0, max=1.0)
                    if mode == "sqrt":
                        shaped = torch.sqrt(excess)
                    elif mode == "quadratic":
                        shaped = excess * excess
                    else:
                        shaped = excess
                    head_values.append((head, text_attention, prior, excess, shaped))

                budget = None
                if mode == "budget":
                    weighted_active = []
                    for _, _, prior, excess, shaped in head_values:
                        weighted_active.append((excess > 0).to(attn_weights.dtype) * prior)
                    budget = torch.sqrt(torch.stack(weighted_active, dim=0).sum(dim=0) + eps)

                for head, text_attention, prior, excess, shaped in head_values:
                    if mode == "budget":
                        strength = gamma * prior * excess / budget
                    else:
                        strength = gamma * prior * shaped
                    strength = torch.clamp(strength, min=0.0, max=1.0).to(attn_weights.dtype)
                    text_attention *= (1.0 - strength)

        if getattr(self.config, "record_intervention_diagnostics", False):
            diag_heads = range(self.num_heads) if getattr(self.config, "record_all_head_diagnostics", False) else head_list
            if diag_heads is not None:
                img_slice = slice(self.config.img_start_pos, self.config.img_start_pos + self.config.img_length)
                text_start_idx = self.config.img_start_pos + self.config.img_length
                threshold = float(getattr(self.config, "adhh_threshold", 0.0))
                soft_temperature = max(float(getattr(self.config, "soft_temperature", 0.05)), 1e-6)
                soft_gamma = float(getattr(self.config, "soft_gamma", 0.5))
                priors = getattr(self.config, "head_attribution_priors", {})
                diagnostic_output_start = int(getattr(self.config, "diagnostic_output_start_pos", text_start_idx))
                diagnostic_recent_window = int(getattr(self.config, "diagnostic_recent_window", 16))
                records = getattr(self.config, "intervention_diagnostics", None)
                if records is not None:
                    for head in diag_heads:
                        key = f"{int(self.layer_idx)}:{int(head)}"
                        output_start = min(max(diagnostic_output_start, text_start_idx), kv_seq_len)
                        recent_start = min(max(kv_seq_len - diagnostic_recent_window, output_start), kv_seq_len)
                        text_slice = slice(text_start_idx, kv_seq_len)
                        question_slice = slice(text_start_idx, output_start)
                        output_slice = slice(output_start, kv_seq_len)
                        recent_slice = slice(recent_start, kv_seq_len)

                        head_weights = attn_weights[:, head, -1, :]
                        head_values = value_states[:, head, :, :]
                        text_attention = head_weights[:, text_slice]
                        question_attention = head_weights[:, question_slice]
                        output_attention = head_weights[:, output_slice]
                        recent_attention = head_weights[:, recent_slice]
                        img_attention = head_weights[:, img_slice]

                        text_mass = torch.sum(text_attention).detach().float().cpu().item()
                        question_mass = torch.sum(question_attention).detach().float().cpu().item()
                        output_mass = torch.sum(output_attention).detach().float().cpu().item()
                        recent_mass = torch.sum(recent_attention).detach().float().cpu().item()
                        img_mass = torch.sum(img_attention).detach().float().cpu().item()
                        img_distribution = img_attention.float() / (torch.sum(img_attention.float(), dim=-1, keepdim=True) + 1e-6)
                        img_entropy = -torch.sum(
                            img_distribution * torch.log(img_distribution + 1e-6),
                            dim=-1,
                        ).detach().float().cpu().item()
                        img_entropy_norm = img_entropy / math.log(max(self.config.img_length, 2))
                        text_value = torch.bmm(text_attention.float().unsqueeze(1), head_values[:, text_slice, :].float()).squeeze(1)
                        img_value = torch.bmm(img_attention.float().unsqueeze(1), head_values[:, img_slice, :].float()).squeeze(1)
                        question_value = torch.bmm(question_attention.float().unsqueeze(1), head_values[:, question_slice, :].float()).squeeze(1)
                        output_value = torch.bmm(output_attention.float().unsqueeze(1), head_values[:, output_slice, :].float()).squeeze(1)
                        recent_value = torch.bmm(recent_attention.float().unsqueeze(1), head_values[:, recent_slice, :].float()).squeeze(1)
                        text_value_norm = torch.linalg.vector_norm(text_value).detach().float().cpu().item()
                        img_value_norm = torch.linalg.vector_norm(img_value).detach().float().cpu().item()
                        question_value_norm = torch.linalg.vector_norm(question_value).detach().float().cpu().item()
                        output_value_norm = torch.linalg.vector_norm(output_value).detach().float().cpu().item()
                        recent_value_norm = torch.linalg.vector_norm(recent_value).detach().float().cpu().item()
                        text_img_value_dot = torch.sum(text_value.float() * img_value.float()).detach().float().cpu().item()
                        text_img_value_cosine = text_img_value_dot / (text_value_norm * img_value_norm + 1e-6)
                        text_img_value_abs_cosine = abs(text_img_value_cosine)
                        img_unit = img_value.float() / (
                            torch.linalg.vector_norm(img_value.float(), dim=-1, keepdim=True) + 1e-6
                        )
                        parallel_coeff = torch.sum(text_value.float() * img_unit, dim=-1, keepdim=True)
                        supported_text_value = torch.clamp(parallel_coeff, min=0.0) * img_unit
                        unsupported_text_value = text_value.float() - supported_text_value
                        supported_text_value_norm = torch.linalg.vector_norm(
                            supported_text_value
                        ).detach().float().cpu().item()
                        unsupported_text_value_norm = torch.linalg.vector_norm(
                            unsupported_text_value
                        ).detach().float().cpu().item()
                        unsupported_text_value_ratio = unsupported_text_value_norm / (text_value_norm + 1e-6)
                        unsupported_total_value_ratio = unsupported_text_value_norm / (
                            text_value_norm + img_value_norm + 1e-6
                        )
                        visual_mass_ratio = img_mass / (img_mass + text_mass + 1e-6)
                        visual_value_ratio = img_value_norm / (img_value_norm + text_value_norm + 1e-6)
                        recent_output_ratio = recent_mass / (text_mass + 1e-6)
                        soft_alpha = torch.sigmoid(torch.tensor((text_mass - threshold) / soft_temperature)).item()
                        records.append({
                            "layer": int(self.layer_idx) if self.layer_idx is not None else -1,
                            "head": int(head),
                            "head_key": key,
                            "attribution_prior": float(priors.get(key, 1.0)),
                            "text_mass": text_mass,
                            "question_attention": question_mass,
                            "output_attention": output_mass,
                            "recent_output_attention": recent_mass,
                            "img_mass": img_mass,
                            "visual_mass_ratio": float(visual_mass_ratio),
                            "img_entropy": float(img_entropy),
                            "img_entropy_norm": float(img_entropy_norm),
                            "text_ratio": float(text_mass / (text_mass + img_mass + 1e-6)),
                            "text_ratio_img_entropy": float((text_mass / (text_mass + img_mass + 1e-6)) * img_entropy_norm),
                            "text_value_norm": text_value_norm,
                            "img_value_norm": img_value_norm,
                            "text_img_value_dot": float(text_img_value_dot),
                            "text_img_value_cosine": float(text_img_value_cosine),
                            "text_img_value_abs_cosine": float(text_img_value_abs_cosine),
                            "text_img_value_orthogonality": float(1.0 - text_img_value_abs_cosine),
                            "supported_text_value_norm": float(supported_text_value_norm),
                            "unsupported_text_value_norm": float(unsupported_text_value_norm),
                            "unsupported_text_value_ratio": float(unsupported_text_value_ratio),
                            "unsupported_total_value_ratio": float(unsupported_total_value_ratio),
                            "visual_value_ratio": float(visual_value_ratio),
                            "recent_output_ratio": float(recent_output_ratio),
                            "removed_text_value_norm": text_value_norm,
                            "question_value_norm": question_value_norm,
                            "output_value_norm": output_value_norm,
                            "recent_output_value_norm": recent_value_norm,
                            "margin": text_mass - threshold,
                            "text_img_log_ratio": math.log((text_mass + 1e-6) / (img_mass + 1e-6)),
                            "hard_trigger": bool(text_mass >= threshold),
                            "soft_alpha": float(soft_alpha),
                            "soft_strength": float(soft_gamma * soft_alpha),
                            "q_len": int(q_len),
                            "kv_seq_len": int(kv_seq_len),
                        })

        # TODO: add attention reweighting code here
        if getattr(self.config, "reweight_text", False) and head_list is not None:
            text_start_idx = self.config.img_start_pos + self.config.img_length
            for head in head_list:
                attn_weights[:, head, :, text_start_idx:] *= self.config.reweight_alpha

        # TODO: add attention reweighting code here
        if getattr(self.config, "reweight_img", False) and head_list is not None:
            img_slice = slice(self.config.img_start_pos, self.config.img_start_pos + self.config.img_length)
            for head in head_list:
                attn_weights[:, head, :, img_slice] *= self.config.reweight_alpha

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        _record_head_output_diagnostics(self.config, self.layer_idx, attn_output, self.num_heads)
        attn_output = _apply_head_output_direction_projection(self.config, self.layer_idx, attn_output, self.num_heads)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        # TODO: add attention statistics calculation code here
        if output_attention_statistics and head_list is not None:   
            attn_statistics = calculate_attention_statistics(attn_weights, labels, head_list, loss=getattr(self.config, "attention_loss", None))
        else:
            attn_statistics = (None, None, None, None)
                
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, *attn_statistics


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        output_attention_statistics: bool = False, # TODO: add output attention statistics indicator
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
        head_list: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        original_query_states = query_states.clone() if getattr(self.config, "record_query_projection_diagnostics", False) else None
        _record_query_diagnostics(self.config, self.layer_idx, query_states, self.num_heads)
        query_states = _apply_query_direction_projection(self.config, self.layer_idx, query_states, self.num_heads)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if original_query_states is not None:
            original_query_states, _ = apply_rotary_pos_emb(original_query_states, key_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _record_query_attention_projection_diagnostics(
            self.config,
            self.layer_idx,
            original_query_states,
            query_states,
            key_states,
            attention_mask,
            self.num_heads,
            self.head_dim,
        )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: add attention statistics calculation code here
        if output_attention_statistics and head_list is not None:   
            attn_weights = calculate_attention_weights(query_states, key_states, attention_mask, self.num_heads, self.head_dim)
            attn_statistics = calculate_attention_statistics(attn_weights, labels, head_list, loss=getattr(self.config, "attention_loss", None))
            del attn_weights
        else:
            attn_statistics = (None, None, None, None)

        dropout_rate = self.attention_dropout if self.training else 0.0
        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )
        attn_output_heads = attn_output.transpose(1, 2).contiguous()
        _record_head_output_diagnostics(self.config, self.layer_idx, attn_output_heads, self.num_heads)
        attn_output_heads = _apply_head_output_direction_projection(
            self.config,
            self.layer_idx,
            attn_output_heads,
            self.num_heads,
        )
        attn_output = attn_output_heads.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, *attn_statistics

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        output_attention_statistics: bool = False,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
        head_list: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                labels=labels,
                head_list=head_list
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        original_query_states = query_states.clone() if getattr(self.config, "record_query_projection_diagnostics", False) else None
        _record_query_diagnostics(self.config, self.layer_idx, query_states, self.num_heads)
        query_states = _apply_query_direction_projection(self.config, self.layer_idx, query_states, self.num_heads)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if original_query_states is not None:
            original_query_states, _ = apply_rotary_pos_emb(original_query_states, key_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        _record_query_attention_projection_diagnostics(
            self.config,
            self.layer_idx,
            original_query_states,
            query_states,
            key_states,
            attention_mask,
            self.num_heads,
            self.head_dim,
        )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # TODO: add attention statistics calculation code here
        if output_attention_statistics and head_list is not None:   
            attn_weights = calculate_attention_weights(query_states, key_states, attention_mask, self.num_heads, self.head_dim)
            attn_statistics = calculate_attention_statistics(attn_weights, labels, head_list, loss=getattr(self.config, "attention_loss", None))
            del attn_weights
        else:
            attn_statistics = (None, None, None, None)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        _record_head_output_diagnostics(self.config, self.layer_idx, attn_output, self.num_heads)
        attn_output = _apply_head_output_direction_projection(self.config, self.layer_idx, attn_output, self.num_heads)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, *attn_statistics


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_attention_statistics: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        labels: Optional[torch.LongTensor] = None,
        head_list: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        _record_residual_diagnostics(self.config, self.layer_idx, hidden_states)
        hidden_states = _apply_residual_direction_projection(self.config, self.layer_idx, hidden_states)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value, *attn_statistics = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_attention_statistics=output_attention_statistics,
            use_cache=use_cache,
            labels=labels,
            head_list=head_list,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)

        if output_attention_statistics:
            outputs += tuple(attn_statistics)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_attention_statistics: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # TODO: get hallucination head list of each layer here
        hal_head_map = {}
        hal_head_list = getattr(self.config, "hal_attention_heads", None)
        if hal_head_list is not None:
            for (layer_idx, head_idx) in hal_head_list:
                if layer_idx not in hal_head_map:
                    hal_head_map[layer_idx] = []
                hal_head_map[layer_idx].append(head_idx)
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_attn_statistics = () if output_attention_statistics else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if layer_idx in hal_head_map:
                head_list = hal_head_map[layer_idx] 
            else:
                head_list = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_attention_statistics,
                    use_cache,
                    labels,
                    head_list,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_attention_statistics=output_attention_statistics,
                    use_cache=use_cache,
                    labels=labels,
                    head_list=head_list,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_attention_statistics:
                all_attn_statistics += (layer_outputs[-4:],)

        # import pdb; pdb.set_trace()
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        if output_attention_statistics:
            outputs = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_attn_statistics,
            )
        else:
            outputs = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        return outputs


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_attention_statistics: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        early_exit_layers: Optional[List[int]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_attention_statistics=output_attention_statistics,
            output_hidden_states=output_hidden_states or early_exit_layers is not None,
            return_dict=return_dict,
            labels=labels,
        )

        if early_exit_layers is not None:
            logits_dict = {}
            # loss_dict = {}
            for i, early_exit_layer in enumerate(early_exit_layers):
                logits = self.lm_head(outputs.hidden_states[early_exit_layer])
                logits_dict[early_exit_layer] = logits
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                # loss_dict[early_exit_layer] = loss
                
            final_outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            return logits_dict, final_outputs
        else:
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
