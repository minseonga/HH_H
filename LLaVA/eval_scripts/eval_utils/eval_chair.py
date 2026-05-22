import os
import argparse
import json
import chair
import numpy as np
import transformers
from eval_bleu import Evaluator
from pycocotools.coco import COCO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--annotation-dir", type=str, default="answer.jsonl")
    parser.add_argument("--caption_file", type=str, default="captions_train2014.json")
    args = parser.parse_args()

    answers = []
    skipped_blank = 0
    with open(args.answers_file) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                skipped_blank += 1
                continue
            try:
                ans = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {args.answers_file} at line {line_number}: {line[:120]!r}"
                ) from exc
            answer = {
                 "caption": ans['text'],
                 "image_id": ans['question_id'],
                 "image": ans['image'],
            }
            answers.append(answer)
    if skipped_blank:
        print(f"[warn] skipped {skipped_blank} blank lines in {args.answers_file}")
    if not answers:
        raise ValueError(f"No valid answers found in {args.answers_file}")

    imids = [answer['image_id'] for answer in answers]

    # initialize CHAIR with generated captions and annotations
    evaluator = chair.CHAIR(imids, args.annotation_dir)
    evaluator.get_annotations()
    
    # compute chair metrics
    cap_dict = evaluator.compute_chair(answers)
    caption_result = cap_dict["sentences"]
    outputs = [i["caption"] for i in caption_result]

    # get gt captions
    caption_file_path = os.path.join(args.annotation_dir, args.caption_file)
    coco = COCO(caption_file_path)
    sampled_img_ids = [i["image_id"] for i in caption_result]
    captions = []
    for sampled_img_id in sampled_img_ids:
        ann_ids = coco.getAnnIds(imgIds=sampled_img_id)
        annotations = coco.loadAnns(ann_ids)
        captions.append([ann['caption'] for ann in annotations])

    golden_reference = []
    candidate_reference = []
    for i, caption in enumerate(captions):
        candidate_reference.append(outputs[i])
        golden_reference.append(caption)

    golden_reference = {k: [{'caption': x} for x in v] for k, v in enumerate(golden_reference)}
    candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}

    evaluator = Evaluator()
    evaluator.do_the_thing(golden_reference, candidate_reference)

    results = evaluator.evaluation_report
    cap_dict['overall_metrics']['Bleu'] = results['Bleu']
    cap_dict['overall_metrics']['avg_caption_length'] = float(
        np.mean([len(output.split()) for output in outputs])
    )

    # save to json pretty print
    chair_json_path = args.answers_file.replace('.jsonl', '_eval_results.json')
    assert chair_json_path != args.answers_file
    
    with open(chair_json_path, "w") as f:
        json.dump(cap_dict, f, indent=4)

    print(cap_dict['overall_metrics'])
