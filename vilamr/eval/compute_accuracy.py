import random
import json
import argparse

from vilamr.utils import load_json, load_jsonl
from vilamr.vilamr_utils import *
from vilamr.eval.extract_answer import extract_answer_mc


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        # return -1
        return random.choice(range(len(choices)))


def main(args):
    print(f"\033[40;32mEvaluation File: {args.result_file}\033[0m")

    ann_raw = load_json(args.ann_file)
    ann_raw_qid = {ann['question_id']: ann for ann in ann_raw}

    predictions = load_jsonl(args.result_file)
    predictions = {pred['question_id']: pred for pred in predictions}

    if len(predictions) != len(ann_raw_qid):
        print(f"\033[40;31mWarning: len(predictions) != len(test data)\033[0m")
        print(f"\033[40;31mlen(predictions) = {len(predictions)}\033[0m")
        print(f"\033[40;31mlen(test data) = {len(ann_raw_qid)}\033[0m")

    else:
        results_true, results_false = [], []
        for question_id, datum in ann_raw_qid.items():
            if question_id not in predictions:
                pred = {'response': 'FAILED', 'prompt': 'Unknown'}
                pred_response = 'FAILED'
            else:
                pred = predictions[question_id]
                pred_response = pred['response']

            # FIXME: extracting answer from generated cot
            responses = pred_response.split("ANSWER:")
            if len(responses) >= 2:
                rationale, answer_text = responses[0].strip(), responses[1].strip()
            else:
                answer_text = responses[0].strip()
                rationale = ""

            if answer_text in OPTIONS:
                answer_letter = answer_text
                pred_idx = get_pred_idx(answer_letter, datum['choice'], OPTIONS)
            elif (len(answer_text) >= 2 and answer_text[0] in OPTIONS
                  and answer_text[1] == "."):  # e.g., 'A.'
                answer_letter = answer_text[0]
                pred_idx = get_pred_idx(answer_letter, datum['choice'], OPTIONS)
            else:
                pred_idx = extract_answer_mc(answer_text, datum['choice'], OPTIONS)
                answer_letter = OPTIONS[pred_idx]

            analysis = {
                'question_id': question_id,
                'ground_truth': OPTIONS[datum['answer']],
                'parsed_ans': answer_letter,
                'input': pred['prompt'],
                'response': pred_response,
            }

            if pred_idx == datum['answer']:
                analysis['is_correct'] = True
                results_true.append(analysis)
            else:
                analysis['is_correct'] = False
                results_false.append(analysis)

        # compute overall acc
        overall_acc = 100 * len(results_true) / len(ann_raw_qid)
        scores = {
            "overall_acc": f"{overall_acc:.1f}"
        }

        print(scores)
        print(f"\033[40;32m{'#' * 120}\033[0m")

        output = {
            "scores": scores,
            "results": {
                "correct": results_true,
                "incorrect": results_false
            }
        }

        output_file = f"{args.result_file[:-6]}_result.jsonl"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_file', type=str,
        # default=f"playground/checkpoints/vilamr-vicuna-13b/result/vilamr-vicuna-13b_pred.jsonl"
        # default=f"experiments/240818_vilamr_8b/result/vilamr-vicuna-13b_pred.jsonl"
        # default=f"experiments/240818_vilamr_8b/result/vilamr-vicuna-13b_pred.jsonl"
        default=f"experiments/240818_vilamr_8b/result/geoqa-cot_mr_rat-merge.jsonl"
    )
    parser.add_argument(
        '--ann_file', type=str,
        # default="playground/data/vilamr_eval/m3cot/test.json",
        # default="playground/data/vilamr_eval/gd_vcr/val.json",
        default="playground/data/vilamr_eval/geoqa/test.json",
    )

    args = parser.parse_args()
    main(args)
