import pandas as pd
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import mojimoji
import MeCab
from utils import save_as_pickle, tag2id, model_path_map

random.seed(123)


def generate_bert_data(all_ginza_df, tokenizer):
    df_list = []
    for file_id in tqdm(all_ginza_df.file_id.unique()):
        ginza_df = all_ginza_df[all_ginza_df.file_id == file_id].copy()
        tag_token_list = ginza_df[["tag", "token"]].apply(tuple, axis=1).tolist()
        token_map = []
        for i, (tag, token) in enumerate(tag_token_list):
            ids = tokenizer.encode(token, add_special_tokens=False)
            tokens = tokenizer.tokenize(token)
            for ginza_bert_idx, (idx, bert_token) in enumerate(zip(ids, tokens)):
                token_map.append((i, tag, token, bert_token, idx, ginza_bert_idx, file_id))
        token_map_df = pd.DataFrame(token_map)
        token_map_df.columns = ["token_id", "tag", "token", "bert_tokens", "bert_id", "ginza_bert_idx", "file_id"]
        df_list.append(token_map_df)
    return pd.concat(df_list)


def get_tag_ids(tags):
    tag_ids = [tag2id[tag] for tag in tags]
    tag_ids = [-100] + tag_ids
    tag_ids = tag_ids + [-100] * (512 - len(tag_ids))
    return tag_ids


def get_mask_ids(ginza_bert_idx_list):
    mask = []
    for idx in ginza_bert_idx_list:
        if idx == 0:
            mask.append(1)
        else:
            mask.append(0)
    mask = [0] + mask
    mask = mask + [0] * (512 - len(mask))
    return mask


def pad_ids(ids):
    ids = [2] + ids + [3]
    ids = ids + [0] * (512 - len(ids))
    return ids


def process_data(ids, tags, ginza_bert_idx_list):
    ids = pad_ids(ids)
    labels = get_tag_ids(tags)
    mask = get_mask_ids(ginza_bert_idx_list)
    data = {
        "ids": ids,
        "labels": labels,
        "mask": mask
    }
    return data


def cut_by_sent(tag_token_list):
    max_length = 510
    sent_list = []
    ids = []
    tags = []
    ginza_bert_idx_list = []
    for i, (tag, token, bert_id, ginza_bert_idx) in enumerate(tag_token_list):
        ids.append(bert_id)
        tags.append(tag)
        ginza_bert_idx_list.append(ginza_bert_idx)
        if token == "。" or len(ids) >= max_length or i == len(tag_token_list) - 1:
            sent_list.append([len(ids), ids, tags, ginza_bert_idx_list])
            ids = []
            tags = []
            ginza_bert_idx_list = []
    return sent_list


def get_data_dict(sent_list):
    max_length = 510
    data_list = []
    ids = []
    tags = []
    ginza_bert_idx_list = []
    for i, sent in enumerate(sent_list):
        total_length = sent[0] + len(ids)
        if total_length >= max_length or i == len(sent_list) - 1:
            data_list.append(process_data(ids, tags, ginza_bert_idx_list))
            ids = sent[1].copy()
            tags = sent[2].copy()
            ginza_bert_idx_list = sent[3].copy()
        else:
            ids.extend(sent[1])
            tags.extend(sent[2])
            ginza_bert_idx_list.extend(sent[3])
            if total_length == max_length:
                data_list.append(process_data(ids, tags, ginza_bert_idx_list))
                ids = []
                tags = []
                ginza_bert_idx_list = []
        if i == len(sent_list) - 1:
            data_list.append(process_data(ids, tags, ginza_bert_idx_list))

    return data_list


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def seperate_sent_list(sent_list, change):
    result_list = [sent_list[:change]]
    remain = sent_list[change:]
    result_list.extend(split(remain, change))
    return result_list


def get_bert_data(df, valid=False, change=None):
    tag_token_list = df[["tag", "token", "bert_id", "ginza_bert_idx"]].apply(tuple, axis=1).tolist()
    sent_list = cut_by_sent(tag_token_list)
    if change:
        data_list = []
        for ls in seperate_sent_list(sent_list, change):
            data_list.extend(get_data_dict(ls))
    else:
        data_list = get_data_dict(sent_list)
    return data_list


def transfer_bert_data_to_dict(bert_data, file_ids, valid=False, change=None):
    data_list = []
    for file_id in tqdm(file_ids):
        df = bert_data[bert_data.file_id == file_id]
        data_list.extend(get_bert_data(df, valid, change))
    return data_list


class NICTokenizer:
    def __init__(self, model_name):
        model_path = model_path_map[model_name]
        self.tagger_jumandic = MeCab.Tagger("-Owakati -d /var/lib/mecab/dic/juman-utf8/")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def parse(self, token):
        token = mojimoji.han_to_zen(token).replace("\u3000", " ")
        return self.tagger_jumandic.parse(token).rstrip("\n")

    def tokenize(self, token):
        result = self.tokenizer.tokenize(self.parse(token))
        if '[UNK]' in result:
            return self.tokenizer.tokenize(" ".join([char for char in token]))
        return result

    def encode(self, token, add_special_tokens=False):
        tokens = self.tokenize(token)
        return self.tokenizer.convert_tokens_to_ids(tokens)


def generate_test_data(model_name):
    print(model_name)
    if model_name.startswith("NICT"):
        tokenizer = NICTokenizer(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path_map[model_name])
    test_token = pd.read_csv("../data/input/test_token.csv").dropna()
    print("Generating bert data")
    bert_test_data = generate_bert_data(test_token, tokenizer)
    bert_test_data['tag'] = "O"
    bert_test_data.loc[bert_test_data["ginza_bert_idx"] != 0, "tag"] = "mask"
    bert_test_data = bert_test_data[['token_id', 'tag', 'token', 'file_id', 'bert_tokens', 'bert_id', 'ginza_bert_idx']]
    print("Writing dict to file")
    file_ids = bert_test_data.file_id.unique()
    test_bert_data_dict = transfer_bert_data_to_dict(bert_test_data, file_ids, valid=True)
    path = f"../data/preprocessed/test_bert_data_dict_{model_name}.pk"
    save_as_pickle(path, test_bert_data_dict)
    return test_bert_data_dict


def check(test_data):
    total = 0
    for d in test_data:
        l = d["labels"]
        total += len([i for i in l if i >= 0])
    assert total == 670092


if __name__ == '__main__':
    for model_name in model_path_map.keys():
        generate_test_data(model_name)
