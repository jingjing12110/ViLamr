import re
import random
import numpy as np

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_answer_mc(response, options, option_inds):
    patterns = [
        # "^" 表示字符串的开头，"$" 表示字符串的结尾。
        r'\(([A-Za-z])\)',  # (B)
        # \. 表示匹配字符 "." , . 是正则表达式中的特殊元字符，匹配除换行符外的任意字符
        r'\(([A-Za-z])\)\.',  # "(B)."
        r'\(([A-Za-z])\):',  # "(B):"
        r'\(([A-Za-z])\) [\s\S]+$',  # (B) XXXXX."
        r'^([A-Za-z])$',  # "b", "B"
        # \b 表示单词边界，
        r'\b([A-Za-z])\b',  # B
        r'\b([A-Za-z])\.',  # B.
        r'\b([A-Za-z]):',  # B:
    ]

    # have "X" in the output
    for p in patterns:
        pattern = re.compile(p, re.IGNORECASE)
        res = pattern.findall(response)
        if len(res) > 0:
            pred = res[0].upper()  # e.g., "B"
            if pred in option_inds:
                ind = option_inds.index(pred)  # 1
                if ind >= len(options):
                    ind = random.choice(range(len(options)))
                # prediction = options[ind]
                return ind

    # find the most similar options
    scores = [score_string_similarity(x, response) for x in options]
    max_idx = int(np.argmax(scores))
    # prediction = options[max_idx]
    return max_idx

