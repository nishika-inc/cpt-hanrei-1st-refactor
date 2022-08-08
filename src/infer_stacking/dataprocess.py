import random
from utils import load_pickle, seed_map, tag2id, save_as_pickle
from post_processing import correct_idx, get_voted_result
from seqeval.metrics import classification_report,f1_score
import pandas as pd
import torch
from sklearn.model_selection import KFold

GINZA_DATA_PATH = ""
MODEL_NAME_LIST = ["cl-charwom", "cl-wom", "cl", "NICT-100k", "NICT-32k", "cl-char"]
TEST_TOKENS = pd.read_csv("../data/input/test_token.csv").dropna().token.tolist()
GINZA_TRAIN_DF = ""
FILE_IDS = ""



class Result:

    def __init__(self, trial_name, model_name, fold, data_type):
        self.trail_name = trial_name
        self.model_name = model_name
        self.fold = fold
        self.data_type = data_type

    @staticmethod
    def _extract_f1_from_report(report):
        return float(report.split()[report.split().index("micro") + 4])

    def retrieve_results(self, path=None):
        if not path:
            path = f"../save/train/{self.trail_name}_{self.model_name}/output/{self.data_type}_pred_{self.fold}.pk"
        self.tags, self.labels, self.logits, report = load_pickle(path)
        self.f1 = self._extract_f1_from_report(report)


class ModelResult:

    def __init__(self, trail_name, model_name, data_type):
        self.trial_name = trail_name
        self.model_name = model_name
        self.data_type = data_type
        self.tokens = self.get_tokens()
        self.fold_results = self.retrieve_results_for_each_fold()

        self.tags = self.get_tags()
        self.tags_corrected = correct_idx(self.tags, self.tokens)
        self.logits = self.get_logits()

        if data_type == "valid":
            self.index = self.get_index()
            self.labels = self.get_labels()
            self.f1 = f1_score([self.labels], [self.tags])
            self.f1_corrected = f1_score([self.labels], [self.tags_corrected])

    def get_tokens(self):
        if self.data_type == "valid":
            return GINZA_TRAIN_DF.token.tolist()
        return TEST_TOKENS

    def retrieve_results_for_each_fold(self, folds=5):
        fold_results = []
        for fold in range(folds):
            result = Result(self.trial_name,
                            self.model_name,
                            fold,
                            self.data_type)
            result.retrieve_results()
            fold_results.append(result)
        return fold_results

    def get_file_ids(self):
        file_ids = FILE_IDS.copy()
        seed = seed_map[self.model_name]
        random.Random(seed).shuffle(file_ids)
        return file_ids

    def get_index(self):
        file_ids = self.get_file_ids()
        df_list = []
        for _, valid_idx in KFold(n_splits=5).split(file_ids):
            valid_file_ids = file_ids[valid_idx]
            for ids in valid_file_ids:
                df_list.append(GINZA_TRAIN_DF[GINZA_TRAIN_DF.file_id == ids])
        concated = pd.concat(df_list)
        index = concated.index
        return index

    def get_from_all_fold(self, attr):
        item_list = []
        for fold in range(5):
            result = self.fold_results[fold]
            item = getattr(result, attr)
            if self.data_type == "valid":
                item_list.extend(item)
            else:
                item_list.append(item)
        return item_list

    def get_tags(self):
        tags = self.get_from_all_fold("tags")
        if self.data_type == "valid":
            return self.sort_according_idx(tags)
        else:
            return get_voted_result(tags)

    def get_labels(self):
        labels = self.get_from_all_fold("labels")
        return self.sort_according_idx(labels)

    def get_logits(self):
        logits = self.get_from_all_fold("logits")
        if self.data_type == "valid":
            logits = self.sort_according_idx(logits)
        return torch.stack(logits, 0)

    def sort_according_idx(self, X):
        return [x for _, x in sorted(zip(self.index, X))]


class TrailResult:

    def __init__(self, trail_name, data_type):
        self.trail_name = trail_name
        self.data_type = data_type
        self.tokens = self.get_tokens()

        print("loading result files")
        self.model_result_dict = {
            model_name: ModelResult(self.trail_name, model_name, data_type)
            for model_name in MODEL_NAME_LIST
        }
        self.tag_list = [
            model_result.tags for model_result in self.model_result_dict.values()
        ]
        self.tag_list_corrected = [
            model_result.tags_corrected for model_result in self.model_result_dict.values()
        ]

    def generate_voted_result(self, tag_list, correct=False):
        voted = get_voted_result(tag_list)
        if correct:
            voted = correct_idx(voted, self.tokens)
        return voted

    def get_tokens(self):
        if self.data_type == "valid":
            return GINZA_TRAIN_DF.token.tolist()
        return TEST_TOKENS

    def generate_voted(self):
        print("generating voted result")
        self.voted = self.generate_voted_result(self.tag_list)
        self.voted_before_corrected = self.generate_voted_result(self.tag_list, correct=True)
        self.voted_after_corrected = self.generate_voted_result(self.tag_list_corrected, correct=True)

    def generate_report(self):
        label_list = GINZA_TRAIN_DF.tag.tolist()
        print("generating report")
        self.f1_report_orig = \
            classification_report([label_list], [self.voted], digits=4)
        self.f1_report_correct = \
            classification_report([label_list], [self.voted_before_corrected], digits=4)
        self.f1_report_vote_after_correct = \
            classification_report([label_list], [self.voted_after_corrected], digits=4)

    def print_all(self):
        print("#" * 50)
        print("f1_orig".ljust(50, '-'))
        for model, result in self.model_result_dict.items():
            print(model, round(result.f1, 4))
        print("f1_dict_corrected".ljust(50, '-'))
        for model, result in self.model_result_dict.items():
            print(model, round(result.f1_corrected, 4))
        print("f1_report_orig".ljust(50, '-'))
        print(self.f1_report_orig)
        print("f1_report_correct".ljust(50, '-'))
        print(self.f1_report_correct)
        print("f1_report_vote_after_correct".ljust(50, '-'))
        print(self.f1_report_vote_after_correct)
        print("#" * 50)


def create_test_data(trail_name):
    test_df = pd.read_csv("../data/input/test_token.csv").dropna().reset_index(drop=True)
    file_ids = test_df.file_id.unique()
    result = TrailResult(trail_name, "test")
    for fold in range(5):
        test_logits = []
        for model in MODEL_NAME_LIST:
            logits_list = result.model_result_dict[model].fold_results[fold].logits
            test_logits.append(logits_list)
        logits = torch.cat(test_logits, axis=-1)
        test_data_list = []
        for file_id in file_ids:
            sub_df = test_df[test_df.file_id == file_id]
            idx = sub_df.index.tolist()
            dic = {"logits": logits[idx],
                   "tokens": sub_df.token.tolist()
                   }
            test_data_list.append(dic)
        path = f"../data/preprocessed/test_stacking_data_{fold}.pk"
        save_as_pickle(path,test_data_list)


def create_mean_test_data():
    test_df = pd.read_csv("../data/input/test_token.csv").dropna().reset_index(drop=True)
    file_ids = test_df.file_id.unique()
    mean_test_data_list = []
    for i in range(len(file_ids)):
        mean_test_data_list.append({'logits':[],'tokens':[]})
    for fold in range(5):
        test_path = f"../data/preprocessed/test_stacking_data_{fold}.pk"
        test_data_list = load_pickle(test_path)
        for i in range(len(file_ids)):
            test_data = test_data_list[i]
            logits = test_data['logits']
            tokens = test_data['tokens']
            mean_test_data_list[i]['logits'].append(logits)
            if fold == 0:
                mean_test_data_list[i]['tokens'] = tokens
    for i in range(len(file_ids)):
        mean_test_data_list[i]['logits'] = torch.mean(torch.stack(mean_test_data_list[i]['logits']), dim=0)
    test_path = "../data/preprocessed/test_stacking_data_mean.pk"
    save_as_pickle(test_path, mean_test_data_list)


if __name__ == '__main__':
    trail_name = "seed_data_for_each_model"
    create_test_data(trail_name)
    create_mean_test_data()

