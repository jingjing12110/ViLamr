OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

VI_MMR_CLASSES = [
    'nature_relation', 'social_relation', 'physical_relation',
    'identity_reasoning', 'function_reasoning', 'physical_property_reasoning',
    'future_prediction'
]
EKnow_CLASSES = ['external_knowledge_reasoning', 'knowledge-based_reasoning']
VI_MMP_CLASSES = [
    'attribute_recognition', 'celebrity_recognition', 'action_recognition',
    'attribute_comparison', 'object_localization', 'spatial_relationship',
    'image_scene', 'image_quality', 'image_style',
    'image_emotion', 'image_topic',
    'ocr'
]
Coarse_VI_MMP_CLASSES = [
    'image_scene', 'image_quality', 'image_style',
    'image_emotion', 'image_topic'
]
FineGrained_VI_MMP_CLASSES = [
    'attribute_recognition', 'celebrity_recognition', 'action_recognition',
    'attribute_comparison', 'object_localization', 'spatial_relationship',
    'ocr'
]

MM_CLASSES = [
    'attribute_recognition', 'celebrity_recognition', 'action_recognition',
    'attribute_comparison', 'object_localization', 'spatial_relationship',
    'image_scene', 'image_quality', 'image_style',
    'image_emotion', 'image_topic',
    'nature_relation', 'social_relation', 'physical_relation',
    'identity_reasoning', 'function_reasoning', 'physical_property_reasoning',
    'external_knowledge_reasoning', 'future_prediction', 'OCR'
]

LLaVA15_TASK_HINT = "Answer with the option's letter from the given choices directly."
answer_formatting_prompt_ft = "answer directly with a single word or number starting with 'ANSWER:'."
answer_formatting_prompt = "give the option's letter for the correct answer in the format 'ANSWER: X'."
CoT_Formatting_Prompts = {
    'mr_detail': f"First perform detailed reasoning based on the given context and then {answer_formatting_prompt}",
    'mr_cot': f"First perform chain-of-thought reasoning based on the given context and then {answer_formatting_prompt}",
    'mr_rat': f"First provide the rationales that arrive at the correct answer to the question and finally {answer_formatting_prompt}",
    'mr_step': f"First provide the intermediate reasoning steps that lead to the correct answer to the question and finally {answer_formatting_prompt}",  # for m3cot
    'mc': LLaVA15_TASK_HINT,
    'none': '',
}


def get_choice_text(choices):
    choice_list = []
    for i, c in enumerate(choices):
        c = c if c != "" else "None"
        choice_list.append(f"{OPTIONS[i]}. {c}")
        # choice_list.append("({}) {}".format(OPTIONS[i], c))
    choice_txt = "\n".join(choice_list)
    return choice_txt


def compute_mm_acc(results):
    vi_mmr_overall = [datum for datum in results if
                      datum['question_type'] in VI_MMR_CLASSES]
    vi_mmr_overall_acc = 100 * len(
        [datum for datum in vi_mmr_overall if datum['is_correct']]
    ) / len(vi_mmr_overall)

    vi_mmr_id = [datum for datum in results if
                 datum['question_type'] == "identity_reasoning"]
    vi_mmr_id_acc = 100 * len(
        [datum for datum in vi_mmr_id if datum['is_correct']]) / len(vi_mmr_id)

    vi_mmr_fc = [datum for datum in results if
                 datum['question_type'] == "function_reasoning"]
    vi_mmr_fc_acc = 100 * len(
        [datum for datum in vi_mmr_fc if datum['is_correct']]) / len(vi_mmr_fc)

    vi_mmr_pp = [datum for datum in results if
                 datum['question_type'] == "physical_property_reasoning"]
    vi_mmr_pp_acc = 100 * len(
        [datum for datum in vi_mmr_pp if datum['is_correct']]) / len(vi_mmr_pp)

    vi_mmr_sr = [datum for datum in results if
                 datum['question_type'] == "social_relation"]
    vi_mmr_sr_acc = 100 * len(
        [datum for datum in vi_mmr_sr if datum['is_correct']]) / len(vi_mmr_sr)

    vi_mmr_nr = [datum for datum in results if
                 datum['question_type'] == "nature_relation"]
    vi_mmr_nr_acc = 100 * len(
        [datum for datum in vi_mmr_nr if datum['is_correct']]) / len(vi_mmr_nr)

    vi_mmr_pr = [datum for datum in results if
                 datum['question_type'] == "physical_relation"]
    vi_mmr_pr_acc = 100 * len(
        [datum for datum in vi_mmr_pr if datum['is_correct']]) / len(vi_mmr_pr)

    vi_mmr_fp = [datum for datum in results if
                 datum['question_type'] == "future_prediction"]
    vi_mmr_fp_acc = 100 * len(
        [datum for datum in vi_mmr_fp if datum['is_correct']]) / len(vi_mmr_fp)

    vi_mmp_overall = [datum for datum in results if
                      datum['question_type'] in VI_MMP_CLASSES]
    vi_mmp_overall_acc = 100 * len(
        [datum for datum in vi_mmp_overall if datum['is_correct']]
    ) / len(vi_mmp_overall)

    know = [datum for datum in results if datum['question_type'] in EKnow_CLASSES]
    know_acc = 100 * len(
        [datum for datum in know if datum['is_correct']]) / len(know)

    total_all_acc = 100 * len(
        [datum for datum in results if datum['is_correct']]) / len(results)
    scores = {
        "total_all": total_all_acc,
        "vi_mmp_all": vi_mmp_overall_acc,
        "vi_mmr_all": vi_mmr_overall_acc,
        "vi_mmr_id": vi_mmr_id_acc,
        "vi_mmr_fc": vi_mmr_fc_acc,
        "vi_mmr_pp": vi_mmr_pp_acc,
        "vi_mmr_sr": vi_mmr_sr_acc,
        "vi_mmr_nr": vi_mmr_nr_acc,
        "vi_mmr_pr": vi_mmr_pr_acc,
        "vi_mmr_fp": vi_mmr_fp_acc,
        "eknow": know_acc,
    }
    return scores


def report_qtype_stats(source_data):
    NR, SR, PR, IR, FR, PP, FP, EKw, Coarse, FineGrained = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for datum in source_data:
        if datum['question_type'] == "nature_relation":
            NR += 1
        elif datum['question_type'] == "social_relation":
            SR += 1
        elif datum['question_type'] == "physical_relation":
            PR += 1
        elif datum['question_type'] == "identity_reasoning":
            IR += 1
        elif datum['question_type'] == "function_reasoning":
            FR += 1
        elif datum['question_type'] == "physical_property_reasoning":
            PP += 1
        elif datum['question_type'] == "future_prediction":
            FP += 1
        elif datum['question_type'] in EKnow_CLASSES:
            EKw += 1
        elif datum['question_type'] in Coarse_VI_MMP_CLASSES:
            Coarse +=1
        elif datum['question_type'] in FineGrained_VI_MMP_CLASSES:
            FineGrained +=1
        else:
            print(datum['question_type'])
    print(f"NR: {NR}, SR: {SR}, PR: {PR}, "
          f"IR: {IR}, FR: {FR}, PP: {PP}, "
          f"FP: {FP}, EKw: {EKw}, "
          f"Coarse: {Coarse}, FineGrained: {FineGrained}")

import re


def find_number_lists_in_string(s):
    # This pattern matches brackets containing comma-separated floats
    pattern = r'\[\s*(?:\d*\.\d+|\d+)(?:\s*,\s*(?:\d*\.\d+|\d+))*\s*\]'
    matches = re.findall(pattern, s)
    # Convert string matches to actual lists of floats
    number_lists = [eval(match) for match in matches]
    return number_lists


def contains_number_list(s):
    # Pattern to match a bracketed list of comma-separated numbers
    pattern = r'\[\s*(?:\d*\.\d+|\d+)(?:\s*,\s*(?:\d*\.\d+|\d+))*\s*\]'
    # Check if there is at least one match in the string
    return bool(re.search(pattern, s))


def contains_letter_bracket(s):
    # Pattern to match a bracketed list of comma-separated numbers
    pattern = r'\(([A-Za-z])\)'
    # Check if there is at least one match in the string
    return bool(re.search(pattern, s))


def contains_image_tags(text):
    # Regular expression to match patterns like <image 1>, <image 2>, etc.
    pattern = r'<image \d+>'
    # Search the text for any match of the pattern
    match_found = bool(re.search(pattern, text))
    return match_found


def convert_choice_string2list(text):
    # 使用正则表达式匹配每行的以字母和冒号开头，后跟任何内容的模式
    pattern = r"^[A-H]:\s*(.+)$"
    results = re.findall(pattern, text, re.MULTILINE)
    return results


def remove_numbered_prefix(lst):
    pattern = r'^[A-Z]\.\s*'  # 匹配以大写字母加点和空格开头的模式
    result = [re.sub(pattern, '', item) for item in lst]
    return result

