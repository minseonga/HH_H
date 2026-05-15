import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import math
from PIL import Image
from transformers import set_seed
import seaborn as sns
import matplotlib.pyplot as plt

from eval_scripts.eval_utils.head_attribution import set_zero_ablation_greedy_search
from eval_scripts.soft_routing.head_prior_utils import load_head_priors
set_zero_ablation_greedy_search()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def json_custom_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Type %s not serializable" % type(obj))


def score_item(difference, layer_idx, head_idx):
    return {
        "layer": int(layer_idx),
        "head": int(head_idx),
        "score": float(difference[layer_idx][head_idx].item()),
    }


def load_candidate_heads(args):
    if not args.candidate_head_path:
        return None
    heads, _, _ = load_head_priors(args.candidate_head_path, top_k=args.candidate_topk)
    return [[int(layer_idx), int(head_idx)] for layer_idx, head_idx in heads]


def unique_preserve_order(values):
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def decode_object_tokens(tokenizer, words):
    tokens = []
    for word in words:
        token_ids = tokenizer(word)['input_ids']
        if len(token_ids) < 2:
            continue
        tokens.append(tokenizer.decode(token_ids[1]))
    return unique_preserve_order(tokens)

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
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )

    questions = []
    sampled_img_ids = []
    with open(os.path.expanduser(args.answers_file), "r") as f:
        caps = json.load(f)["sentences"]
        for cap in caps:
            if cap["metrics"]["CHAIRs"] == 1:
                image_id = "{:012d}".format(cap["image_id"])
                image_file = cap.get("image", f"COCO_{args.image_split}_{image_id}.jpg")
                question = {
                    "question_id": cap["image_id"],
                    "image": image_file,
                    "image_path": os.path.join(args.image_folder, image_file),
                    "text": "Please describe this image in detail.",
                    "caption": cap["caption"],
                    "mscoco_hallucinated_words": [i[0] for i in cap["mscoco_hallucinated_words"]],
                    "mscoco_non_hallucinated_words": [i[0] for i in cap["mscoco_non_hallucinated_words"]],
                }
                questions.append(question)
                sampled_img_ids.append(image_id)
    print(len(questions))

    questions = questions[args.start_idx:args.end_idx]
    sampled_img_ids = sampled_img_ids[args.start_idx:args.end_idx]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    candidate_heads = load_candidate_heads(args)
    if candidate_heads is not None:
        print(f"Using {len(candidate_heads)} candidate heads for attribution: {candidate_heads}")
    os.makedirs(args.output_path, exist_ok=True)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        print(line)
        question_id = line["question_id"]
        image_file = line["image"]
        hall_path = os.path.join(args.output_path, f'hallucination_influences_{question_id}.pth')
        non_hall_path = os.path.join(args.output_path, f'non_hallucination_influences_{question_id}.pth')
        if args.resume and os.path.exists(hall_path) and os.path.exists(non_hall_path):
            print(f"skip existing attribution for {question_id}")
            continue

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        hallucinated_words = unique_preserve_order(line['mscoco_hallucinated_words'])
        non_hallucinated_words = unique_preserve_order(line['mscoco_non_hallucinated_words'])
        hallucinated_tokens = decode_object_tokens(tokenizer, hallucinated_words)
        non_hallucinated_tokens = decode_object_tokens(tokenizer, non_hallucinated_words)

        with torch.inference_mode():
            _, hallucination_influences, non_hallucination_influences = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=True, 
                hallucinated_tokens=hallucinated_tokens, 
                non_hallucinated_tokens=non_hallucinated_tokens,
                candidate_heads=candidate_heads,
                max_hall_attribution_events=args.max_hall_events_per_sample,
                max_nonhall_attribution_events=args.max_nonhall_events_per_sample,
                dedupe_attribution_tokens=args.dedupe_attribution_tokens,
                influence_score=args.influence_score)
            
        torch.save(hallucination_influences, hall_path)
        torch.save(non_hallucination_influences, non_hall_path)
        torch.cuda.empty_cache()

        if args.save_heatmaps:
            influences = []
            for _, v in hallucination_influences.items():
                influence = torch.zeros(args.layer_num, args.head_num)
                for layer_idx in range(args.layer_num):
                    for head_idx in range(args.head_num):
                        influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
                influences.append(influence)
            if influences:
                plt.figure(figsize=(8,8))
                sns.heatmap(torch.mean(torch.stack(influences), 0).cpu().numpy(),cmap="coolwarm", center=0)
                plt.savefig(f'{args.output_path}/hallucination_influences_{question_id}.png')
                plt.close()
            
            influences = []
            for _, v in non_hallucination_influences.items():
                influence = torch.zeros(args.layer_num, args.head_num)
                for layer_idx in range(args.layer_num):
                    for head_idx in range(args.head_num):
                        influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
                influences.append(influence)
            if influences:
                plt.figure(figsize=(8,8))
                sns.heatmap(torch.mean(torch.stack(influences), 0).cpu().numpy(),cmap="coolwarm", center=0)
                plt.savefig(f'{args.output_path}/non_hallucination_influences_{question_id}.png')
                plt.close()
        
    
def get_constrative_influence(args):
    candidate_heads = load_candidate_heads(args)

    files = os.listdir(args.output_path)
    hallucination_samples = []
    non_hallucination_samples = []
    for file in files:
        if file.endswith('pth'):
            if file.startswith('hal'):
                hallucination_sample = torch.load(os.path.join(args.output_path, file))
                influences = []
                for _, v in hallucination_sample.items():
                    influence = torch.zeros(args.layer_num, args.head_num)
                    for layer_idx in range(args.layer_num):
                        for head_idx in range(args.head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
                    influences.append(influence)
                hallucination_samples += influences
            else:
                non_hallucination_sample = torch.load(os.path.join(args.output_path, file))
                influences = []
                for _, v in non_hallucination_sample.items():
                    influence = torch.zeros(args.layer_num, args.head_num)
                    for layer_idx in range(args.layer_num):
                        for head_idx in range(args.head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
                    influences.append(influence)
                non_hallucination_samples += influences

    hallucinated_scores = torch.stack(hallucination_samples)
    non_hallucinated_scores = torch.stack(non_hallucination_samples)

    plt.figure(figsize=(8,8))
    difference = torch.mean(hallucinated_scores,0).float() - torch.mean(non_hallucinated_scores,0).float() 
    ax = sns.heatmap(difference.cpu().numpy(),cmap="coolwarm", center=0)
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    print(f'{args.output_path}/constrastive_influences.png')
    plt.savefig(f'{args.output_path}/constrastive_influences.png')

    results = {}
    if candidate_heads is not None:
        candidate_scores = [score_item(difference, layer_idx, head_idx) for layer_idx, head_idx in candidate_heads]
        hal_scores = sorted(candidate_scores, key=lambda item: item["score"], reverse=True)
        non_hal_scores = sorted(candidate_scores, key=lambda item: item["score"])
        results.update({'hal_heads': [[item["layer"], item["head"]] for item in hal_scores]})
        results.update({'hal_head_scores': hal_scores})
        results.update({'non_hal_heads': [[item["layer"], item["head"]] for item in non_hal_scores]})
        results.update({'non_hal_head_scores': non_hal_scores})
        results.update({'contrastive_scores': candidate_scores})
        results.update({'candidate_head_path': args.candidate_head_path})
    else:
        _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=True)
        indices = [[int(flat_indice.item()) // args.head_num, int(flat_indice.item()) % args.head_num] for flat_indice in flat_indices]
        results.update({'hal_heads': indices})
        results.update({'hal_head_scores': [score_item(difference, layer_idx, head_idx) for layer_idx, head_idx in indices]})
        _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=False)
        indices =  [[int(flat_indice.item()) // args.head_num, int(flat_indice.item()) % args.head_num] for flat_indice in flat_indices]
        results.update({'non_hal_heads': indices})
        results.update({'non_hal_head_scores': [score_item(difference, layer_idx, head_idx) for layer_idx, head_idx in indices]})
        results.update({
            'contrastive_scores': [
                score_item(difference, layer_idx, head_idx)
                for layer_idx in range(args.layer_num)
                for head_idx in range(args.head_num)
            ]
        })
    print(results)

    print(f'{args.output_path}/attribution_result.json')
    with open(f'{args.output_path}/attribution_result.json', 'w') as file:
        json.dump(results, file, default=json_custom_serializer) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true", default=False)
    parser.add_argument("--load-4bit", action="store_true", default=False)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--image-split", type=str, default="train2014")
    parser.add_argument("--annotation-dir", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
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
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--candidate-head-path", type=str, default="")
    parser.add_argument("--candidate-topk", type=int, default=20)
    parser.add_argument("--max-hall-events-per-sample", type=int, default=1)
    parser.add_argument("--max-nonhall-events-per-sample", type=int, default=3)
    parser.add_argument("--dedupe-attribution-tokens", action="store_true", default=True)
    parser.add_argument("--no-dedupe-attribution-tokens", dest="dedupe_attribution_tokens", action="store_false")
    parser.add_argument("--influence_score", type=str, default='prob_diff')
    parser.add_argument("--layer_num", type=int, default=32)
    parser.add_argument("--head_num", type=int, default=32)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--save_heatmaps", action="store_true", default=False)
    args = parser.parse_args()
    set_seed(args.seed)
    
    # get hallucination and non-hallucination influences for each question
    eval_model(args)
    # average the influences over all questions
    get_constrative_influence(args)
