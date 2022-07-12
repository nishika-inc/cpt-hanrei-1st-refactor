import random
from sklearn.model_selection import KFold
from utils import load_pickle, seed_map
from seqeval.metrics import classification_report,f1_score
from collections import Counter
import pandas as pd
import torch
import numpy as np
pd.options.display.max_rows = 10000

def correct_idx(input_tags, tokens):
    tags = [i for i in input_tags]
    valid_idxs = get_seq_idx(tags, True, return_type="idx", flatten=False)
    for idxs in valid_idxs:
        string = "".join([tokens[i] for i in idxs])
        if len(string) <= 2:
            for i in idxs:
                tags[i] = "O"
    invalid_idxs = get_seq_idx(tags, False, return_type="idx", flatten=True)
    for i in invalid_idxs:
        tags[i] = "O"
    return tags


def is_valid_seq(seq):
    return seq[0].split("-")[0] == "B"


def get_seq_idx(tags, valid_flag=True, return_type="idx", flatten=True):
    begin = False
    category = None
    all_idx_ls = []
    all_tag_ls = []
    idx_ls = []
    tag_ls = []
    for i, tag in enumerate(tags):
        if isinstance(tag, str):
            if tag != "O":
                pos = tag.split("-")[0]
                category = tag.split("-")[1]
                if not begin:
                    begin = True
                else:
                    if category != tag_ls[0].split("-")[1]:
                        begin = False
                        if is_valid_seq(tag_ls) == valid_flag:
                            all_idx_ls.append(idx_ls)
                            all_tag_ls.append(tag_ls)
                        idx_ls = []
                        tag_ls = []
                idx_ls.append(i)
                tag_ls.append(tag)
            else:
                if begin:
                    begin = False
                    if is_valid_seq(tag_ls) == valid_flag:
                        all_idx_ls.append(idx_ls)
                        all_tag_ls.append(tag_ls)
                    idx_ls = []
                    tag_ls = []
    if return_type == "idx":
        result = all_idx_ls
    elif return_type == "tag":
        result = all_tag_ls
    if flatten:
        return [i for ls in result for i in ls]
    else:
        return result


def get_voted_result(result_list):
    voted_list = []
    for idx in range(len(result_list[0])):
        ls = [ls[idx] for ls in result_list]
        if len(set(ls)) == 1:
            voted_list.append(ls[0])
        else:
            counter = Counter(ls)
            voted_list.append(counter.most_common(1)[0][0])
    return voted_list