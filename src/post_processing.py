import random

import torch
from sklearn.model_selection import KFold
from utils import load_pickle
from seqeval.metrics import classification_report, f1_score
from collections import Counter
import pandas as pd


# after set all bad index to "O"
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


def get_submission(tag_list):
    sub = pd.read_csv("data/input/sample_submission.csv")
    tokens = pd.read_csv("data/input/test_token.csv")
    tokens = tokens.dropna()
    tokens["tag"] = tag_list
    tokens = tokens.sort_values(["file_id", "token_id"])
    submission = sub.drop("tag", axis=1).merge(tokens[["file_id", "token_id", "tag"]], how="outer")
    return submission


def load_pred(output_dir):
    valid_tags = []
    valid_lables = []
    test_preds = []

    for fold in range(5):
        tags, labels, _ = load_pickle(f"{output_dir}/valid_pred_{fold}.pk")
        valid_tags.extend(tags)
        valid_lables.extend(labels)
        test_tags = load_pickle(f"{output_dir}/test_pred_{fold}.pk")
        test_preds.append(test_tags)
    return valid_tags, valid_lables, test_preds


def post_preprocess(save_dir, output_dir, name):
    valid_tags, valid_lables, test_preds = load_pred(output_dir)

    all_report = classification_report([valid_lables], [valid_tags], digits=4)
    with open(f"{save_dir}/{name}_classify_report_orig.txt", "w") as f:
        f.write(all_report)

    corrected_tags = correct_idx(valid_tags)
    corrected_report = classification_report([valid_lables], [corrected_tags], digits=4)
    with open(f"{save_dir}/{name}_classify_report_corrected.txt", "w") as f:
        f.write(corrected_report)

    volted_result = get_voted_result(test_preds)
    sub_file = get_submission(volted_result)
    sub_file.to_csv(f"{save_dir}/org-{name}.csv", index=False)

    corrected_result = correct_idx(volted_result)
    sub_file = get_submission(corrected_result)
    sub_file.to_csv(f"{save_dir}/cor-{name}.csv", index=False)

    return all_report, corrected_report


BERT_TRAIN_DATA_PATH = f"data/preprocessed/ginza_train_data.csv"
MODEL_NAME_LIST = ["_cl-charwom", "_cl-wom", "_cl", "_NICT-100k", "_NICT-32k", "_cl-char"]


class ResultFetcher:

    def __init__(self, trail_name, topn=4, data_type="valid"):
        self.data_type=data_type
        self.topn =topn
        self.trail_name = trail_name
        self.pred_tags_dict = [{} for i in range(5)]
        self.pred_labels_dict = [{} for i in range(5)]
        self.pred_logits_dict = [{} for i in range(5)]
        self.pred_report_dict = [{} for i in range(5)]
        self.pred_f1_dict = [{} for i in range(5)]
        self.valid_df_list = self.get_valid_df_list()
        self.retrieve_valid_result()
        self.f1_df = pd.DataFrame(self.pred_f1_dict)
        self.best_model_list = self.f1_df.apply(self.get_large_n, axis=1)

    def get_large_n(self,s) -> list:
        return s.abs().nlargest(self.topn).index.tolist()

    @staticmethod
    def read_read_bert_train_data():
        return pd.read_csv(BERT_TRAIN_DATA_PATH)

    @staticmethod
    def get_file_id(bert_train_data):
        file_ids = bert_train_data.file_id.unique()
        random.Random(123).shuffle(file_ids)
        return file_ids

    def get_valid_df_list(self):
        bert_train_data = self.read_read_bert_train_data()
        file_ids = self.get_file_id(bert_train_data)
        valid_df_list = []
        for _, valid_idx in KFold(n_splits=5).split(file_ids):
            df_list = []
            valid_file_ids = file_ids[valid_idx]
            for ids in valid_file_ids:
                df_list.append(bert_train_data[bert_train_data.file_id == ids])
            valid_df = pd.concat(df_list).reset_index(drop=True)
            valid_df_list.append(valid_df)
        return valid_df_list

    @staticmethod
    def _extract_f1_from_report(report):
        return float(report.split()[report.split().index("micro") + 4])

    def retrieve_valid_result(self):
        for fold in range(5):
            for model in MODEL_NAME_LIST:
                pred_tags, pred_labels, pred_logits, report = load_pickle(
                    f"save/train/{self.trail_name}{model}/output/{self.data_type}_pred_{fold}.pk")
                self.pred_tags_dict[fold][model] = pred_tags
                self.pred_labels_dict[fold][model] = pred_labels
                # self.pred_logits_dict[fold][model] = pred_logits
                self.pred_logits_dict[fold][model] = torch.cat(pred_logits)
                self.pred_report_dict[fold][model] = report
                self.pred_f1_dict[fold][model] = self._extract_f1_from_report(report)


class F1Calaulator:

    def __init__(self, result: ResultFetcher):
        self.result = result
        self.all_tokens = self.get_all_tokens()
        self.pred_tag_dict_corrected = self.get_pred_tag_dict_corrected()

        self.label_list = self.get_label_list()
        self.tag_dict_orig = self.get_tag_dict()
        self.tag_dict_corrected = self.get_tag_dict(use_orig=False)

        print("calculate f1")
        self.f1_dict_orig = self.calculate_f1_dict()
        self.f1_dict_corrected = \
            self.calculate_f1_dict(use_orig=False)
        self.f1_avg_orig = \
            self.calculate_item_avg(self.f1_dict_orig)
        self.f1_avg_corrected =\
            self.calculate_item_avg(self.f1_dict_corrected)

        print("generating voted result")
        self.voted = self.genereat_voted_result(list(self.tag_dict_orig.values()))
        self.voted_before_corrected = \
            self.genereat_voted_result(list(self.tag_dict_orig.values()), correct=True)
        self.voted_after_corrected \
            = self.genereat_voted_result(list(self.tag_dict_corrected.values()), correct=True)

        print("generating report")
        self.f1_report_orig = \
            classification_report([self.label_list], [self.voted], digits=4)
        self.f1_report_correct = \
            classification_report([self.label_list], [self.voted_before_corrected], digits=4)
        self.f1_report_vote_after_correct = \
            classification_report([self.label_list], [self.voted_after_corrected], digits=4)

    @staticmethod
    def calculate_item_avg(dic: dict):
        return sum([item for item in dic.values()]) / len(dic)

    def get_pred_tag_dict_corrected(self):
        pred_tag_dict_corrected = [{} for i in range(5)]
        for fold in range(5):
            tokens = self.result.valid_df_list[fold]["token"]
            for model in MODEL_NAME_LIST:
                tags = self.result.pred_tags_dict[fold][model]
                corrected = correct_idx(tags, tokens)
                pred_tag_dict_corrected[fold][model] = corrected
        return pred_tag_dict_corrected

    def get_label_list(self):
        label_list = []
        for fold in range(5):
            label_list.extend(self.result.pred_labels_dict[fold][MODEL_NAME_LIST[0]])
        return label_list

    def get_tag_dict(self, use_orig=True):
        tag_dict = {model: [] for model in MODEL_NAME_LIST}
        for model in MODEL_NAME_LIST:
            for fold in range(5):
                if use_orig:
                    tag_dict[model].extend(self.result.pred_tags_dict[fold][model])
                else:
                    tag_dict[model].extend(self.pred_tag_dict_corrected[fold][model])
        return tag_dict

    def calculate_f1_dict(self, use_orig=True):
        f1_dict = {model: -1 for model in MODEL_NAME_LIST}
        for model in MODEL_NAME_LIST:
            if use_orig:
                tags = self.tag_dict_orig[model]
            else:
                tags = self.tag_dict_corrected[model]
            f1_dict[model] = f1_score([self.label_list], [tags])
        return f1_dict

    def get_all_tokens(self):
        tokens = []
        for df in self.result.valid_df_list:
            tokens.extend(df["token"])
        return tokens

    def genereat_voted_result(self, tag_list, correct=False):
        voted = get_voted_result(tag_list)
        if correct:
            voted = correct_idx(voted, self.all_tokens)
        return voted


    def print_all(self):
        print("#"*50)
        print("f1_dict_orig".ljust(50, '-'))
        print(self.f1_dict_orig)
        print("f1_dict_corrected".ljust(50, '-'))
        print(self.f1_dict_corrected)
        print("f1_avg_orig".ljust(50, '-'))
        print(self.f1_avg_orig)
        print("f1_avg_corrected".ljust(50, '+'))
        print(self.f1_avg_corrected)
        print("f1_report_orig".ljust(50, '-'))
        print(self.f1_report_orig)
        print("f1_report_correct".ljust(50, '-'))
        print(self.f1_report_correct)
        print("f1_report_vote_after_correct".ljust(50, '-'))
        print(self.f1_report_vote_after_correct)
        print("#" * 50)


