import math
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from vilamr.conversation import conv_templates
from vilamr.utils import disable_torch_init
from vilamr.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vilamr.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path
)
from vilamr.model.builder import load_pretrained_model
from vilamr.vilamr_utils import *
from vilamr.utils import load_json, load_jsonl, str2bool

local_rank = None

EVAL_TASK_FILES = {
    "pca": "playground/data/vilamr_eval/pca_bench/test.json",
    "rwqa": "playground/data/vilamr_eval/rwqa/test.json",
    "mmmu": "playground/data/vilamr_eval/mmmu/val.json",
    "mmstar": "playground/data/vilamr_eval/mmstar/test.json",
    "mmb_dev": "playground/data/vilamr_eval/mm_bench/dev_en.json",
    "mmb_test": "playground/data/vilamr_eval/mm_bench/dev_en.json",
    "geoqa": "playground/data/vilamr_eval/geoqa/test.json",
    "sqa_img": "playground/data/vilamr_eval/sqa_img/test_img.json",
    "gd_vcr": "playground/data/vilamr_eval/gd_vcr/val.json",
    "vcr": "playground/data/vilamr_eval/vcr/val.json",
    "m3cot": "playground/data/vilamr_eval/m3cot/test.json",
    "wemath": "playground/data/vilamr_eval/wemath/testmini.json",
}


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def model_inference(
        args, model, tokenizer, image_processor, model_name,
        response_formatting_prompt=""
):
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    print(f"\033[40;32mTest File: {args.question_file}\033[0m")
    print(f"\033[40;32mTotal Sample: {len(questions)}\033[0m")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if args.resume_from_local and os.path.exists(answers_file):
        local_result = load_jsonl(answers_file)
        local_result_qids = [result['question_id'] for result in local_result]

        questions_unprocessed = []
        for question in questions:
            if question['id'] not in local_result_qids:
                questions_unprocessed.append(question)
        print(f"need to be inference: {len(questions_unprocessed)}")

        ans_file = open(answers_file, "a")
    else:
        questions_unprocessed = questions
        ans_file = open(answers_file, "w")

    questions_unprocessed = get_chunk(
        questions_unprocessed, args.num_chunks, args.chunk_idx
    )
    for data in tqdm(questions_unprocessed, ncols=80):
        question = data["question"]
        choice_text = get_choice_text(data["choice"])

        # FIXME textual context
        if 'context' in data and data['context'] is not None:
            qs = f"{DEFAULT_IMAGE_TOKEN}\nContext: {data['context']}\n\n{question}\n{choice_text}"
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{question}\n{choice_text}"

        qs = qs + f'\n{response_formatting_prompt}'
        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        conv.tokenizer = tokenizer

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        # img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        # image
        img_file = os.path.join(args.image_folder, data["img_dir"])
        image = Image.open(img_file).convert('RGB')
        image_tensor = process_images(
            [image], image_processor, model.config
        )[0]
        images = image_tensor.unsqueeze(0).half().cuda()
        image_sizes = [image.size]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                # top_p=args.top_p,
                # num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=attention_mask,
            )

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        ans_file.write(json.dumps({
            "question_id": data["question_id"],
            "prompt": cur_prompt,
            "response": outputs,
            "model_id": model_name,
        }) + "\n")
        ans_file.flush()
    ans_file.close()


def main(args):
    # Load Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"\033[33mmodel_path: {model_path}\nmodel_name: {model_name}\033[0m")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        vision_tower_mix=args.vision_tower_mix,
        # device="cpu",
        # ADD for low-resource debug
        # load_4bit=True if args.for_debug else False,
        # for_debug=args.for_debug,
        # load_8bit=True,  # ADD for low-resource debug
    )
    # model, tokenizer, image_processor = None, None, None

    for cot_prompt_type, task in zip(args.cot_prompt_types, args.eval_tasks):
        args.question_file = EVAL_TASK_FILES[task]

        args.answers_file = f"{model_path}/result/{task}-cot_{cot_prompt_type}-{args.answers_file_id}"
        print(args.answers_file)

        response_format_prompt = CoT_Formatting_Prompts[cot_prompt_type]
        print(f"\033[40;32mCoTP Type <{cot_prompt_type}>: {response_format_prompt}\033[0m")

        model_inference(args, model, tokenizer, image_processor, model_name, response_format_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base", type=str, default=None
    )
    parser.add_argument(
        "--model_path", type=str,
        default="playground/checkpoints/vilamr-llama3-8b",
    )
    parser.add_argument(
        "--vision_tower_mix", default=True, type=str2bool
    )
    parser.add_argument(
        "--eval_tasks", nargs='+',
        # default=[
        #     "pca", "rwqa", "mmmu", "mmstar", "mmb_dev",
        #     "geoqa", "sqa_img", "gd_vcr", "vcr", 'm3cot'
        # ],
        default=['geoqa'],
    )
    parser.add_argument(
        "--cot_prompt_types", nargs='+',
        default=['mr_rat'],
    )
    parser.add_argument(
        "--answers_file_id", type=str, default="0.jsonl"
    )
    parser.add_argument(
        "--for_debug", default=True, type=str2bool)

    parser.add_argument(
        "--resume_from_local", action="store_true")
    parser.add_argument(
        "--image_folder", type=str,
        default="playground/data/eval_images/",
    )
    parser.add_argument(
        "--conv_mode", type=str,
        # default="vicuna_v1",
        default="llava_llama_3_1"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=1)
    parser.add_argument(
        "--chunk-idx", type=int, default=0)
    parser.add_argument(
        "--temperature", type=float,
        default=0,
    )
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--max_new_tokens", type=int,
        # default=1024,
        default=512,
    )
    parser.add_argument(
        "--single-pred-prompt", action="store_true")
    parser.set_defaults(single_pred_prompt=True)

    args = parser.parse_args()
    print(args)

    main(args)
