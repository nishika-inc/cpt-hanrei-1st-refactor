import os.path
from sklearn.model_selection import KFold
import pickle
import pandas as pd
from tqdm.notebook import tqdm
import random
from transformers import AutoTokenizer
import mojimoji

random.seed(123)

ginza_data_path_map = {
    "orig": "data/preprocessed/ginza_train_data.csv",
    "aug": "data/preprocessed/ginza_train_data_aug.csv",
    "aug_2": "data/preprocessed/ginza_train_data_aug_2.csv",
}

model_path_map = {
    "cl-wom": "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-charwom": "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "cl": "cl-tohoku/bert-base-japanese",
    "cl-char": "cl-tohoku/bert-base-japanese-char",
    "NICT-100k": "NICT_BERT-base_JapaneseWikipedia_100K",
    "NICT-32k": "NICT_BERT-base_JapaneseWikipedia_32K_BPE",
    "bert32k": "bert-japanese/BERT-base_mecab-ipadic-bpe-32k_whole-word-mask",
    "cl-bert": "cl-tohoku/bert-base-japanese-whole-word-masking"
}


# import MeCab


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


def save_as_pickle(path, data):
    with open(path, 'wb') as web:
        pickle.dump(data, web)


def edit_tag(tag):
    if tag == "O" or tag.startswith("I"):
        return tag
    else:
        category = tag.split('-')[-1]
        return f"I-{category}"


id2tag = {-100: 'mask',
          0: 'O',
          1: 'B-TIMEX',
          2: 'I-TIMEX',
          3: 'B-PERSON',
          4: 'I-PERSON',
          5: 'B-ORGFACPOS',
          6: 'I-ORGFACPOS',
          7: 'B-LOCATION',
          8: 'I-LOCATION',
          9: 'B-MISC',
          10: 'I-MISC'}

tag2id = {'B-LOCATION': 7,
          'B-MISC': 9,
          'B-ORGFACPOS': 5,
          'B-PERSON': 3,
          'B-TIMEX': 1,
          'I-LOCATION': 8,
          'I-MISC': 10,
          'I-ORGFACPOS': 6,
          'I-PERSON': 4,
          'I-TIMEX': 2,
          'O': 0,
          'mask': -100}


class NICTokenizer:
    def __init__(self, model_name):
        model_path = model_path_map[model_name]
        self.tagger_jumandic = MeCab.Tagger("-Owakati -d /var/lib/mecab/dic/juman-utf8/")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def parse(self, token):
        token = mojimoji.han_to_zen(token).replace("\u3000", " ")
        return self.tagger_jumandic.parse(token).rstrip("\n")

    def tokenize(self, token):
        result = self.tokenizer.tokenize(parse(token))
        if '[UNK]' in result:
            return self.tokenizer.tokenize(" ".join([char for char in token]))
        return result

    def encode(self, token, add_special_tokens=False):
        tokens = self.tokenize(token)
        return self.tokenizer.encode(tokens, add_special_tokens=False)


class DataProcessor:
    def __init__(self, data_name, model_name, change):
        self.data_name = data_name
        self.change = change
        self.model_name = model_name
        self.bert_train_data_path = f"data/preprocessed/bert_train_data_{data_name}_{model_name}.csv"
        self.generate_train_data()

    def generate_train_data(self):
        if not os.path.exists(self.bert_train_data_path):
            print("Generating bert_data")
            self.generate_bert_train_data()
        self.bert_train_data = pd.read_csv(self.bert_train_data_path)
        print("Generating dict data")
        self.generate_dict_data()

    def read_ginza_data(self):
        path = ginza_data_path_map[self.data_name]
        df = pd.read_csv(path).reset_index(drop=True)
        token_length = df.token.apply(len)
        df.loc[token_length > 50, "token"] = df[token_length > 50]["token"].str[0]
        return df

    def generate_bert_train_data(self):
        ginza_train_data = self.read_ginza_data()
        tokenizer = AutoTokenizer.from_pretrained(model_path_map[self.model_name])
        bert_train_data = generate_bert_data(ginza_train_data, tokenizer)
        bert_train_data.loc[bert_train_data["ginza_bert_idx"] != 0, "tag"] = "mask"
        bert_train_data.to_csv(self.bert_train_data_path, index=False)

    def generate_dict_data(self):
        # file_ids = self.bert_train_data.file_id.unique()
        # random.Random(123).shuffle(file_ids)
        # for epoch, (train_idx, valid_idx) in enumerate(KFold(n_splits=5).split(file_ids)):
        #     train_file_ids = file_ids[train_idx]
        #     valid_file_ids = file_ids[valid_idx]
        #     train_path = f"data/preprocessed/train_bert_data_dict_{self.data_name}_{self.model_name}_{epoch}.pk"
        #     valid_path = f"data/preprocessed/valid_bert_data_dict_{self.data_name}_{self.model_name}_{epoch}.pk"
        #     if not os.path.exists(train_path):
        #         train_bert_data_dict = transfer_bert_data_to_dict(self.bert_train_data, train_file_ids)
        #         save_as_pickle(train_path, train_bert_data_dict)
        #     if not os.path.exists(valid_path):
        #         valid_bert_data_dict = transfer_bert_data_to_dict(self.bert_train_data, valid_file_ids, valid=True)
        #         save_as_pickle(valid_path, valid_bert_data_dict)
        train_file_ids = [89332, 89355, 89424, 89271, 89428, 89523, 89407, 89291, 89312, 89322, 89646, 89552, 89383,
                          89642, 89385, 89460, 89270, 89391, 89639, 89282, 89363, 89353, 89449, 89652, 89452, 89486,
                          89491, 89285, 89578, 89462, 89511, 89434, 89548, 89275, 89487, 89321, 89497, 89413, 89508,
                          89417, 89638, 89501, 89357, 89558, 89530, 89471, 89410, 89374, 89626, 89477, 89566, 89512,
                          89284, 89398, 89377, 89277, 89414, 89556, 89362, 89429, 89466, 89375, 89242, 89337, 89373,
                          89615, 89645, 89604, 89396, 89498, 89549, 89286, 89480, 89562, 89561, 89392, 89465, 89612,
                          89648, 89289, 89423, 89647, 89493, 89483, 89372, 89472, 89333, 89409, 89580, 89437, 89422,
                          89435, 89368, 89632, 89650, 89394, 89330, 89455, 89544, 89259, 86910, 89405, 89546, 89369,
                          88850, 89532]
        valid_file_ids = [89447, 89390, 89527, 89226, 89318, 89386, 89366, 89354, 89303, 89300, 89453, 89301, 89293,
                          89278, 89378, 89334, 89442, 89370, 89371, 89489, 89263, 89563, 89533, 89524, 89551, 89614,
                          89264, 89454, 89484, 89404, 89327, 89360, 89653, 89510, 89651, 89636]
        train_path = f"data/preprocessed/train_bert_data_dict_{self.data_name}_{self.model_name}.pk"
        valid_path = f"data/preprocessed/valid_bert_data_dict_{self.data_name}_{self.model_name}.pk"
        if not os.path.exists(train_path):
            train_bert_data_dict = transfer_bert_data_to_dict(self.bert_train_data, train_file_ids)
            save_as_pickle(train_path, train_bert_data_dict)
        if not os.path.exists(valid_path):
            valid_bert_data_dict = transfer_bert_data_to_dict(self.bert_train_data, valid_file_ids, True,self.change)
            save_as_pickle(valid_path, valid_bert_data_dict)


def generate_test_data(model_name):
    print(model_name)
    if model_name.startswith("NICT"):
        tokenizer = NICTokenizer(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path_map[model_name])

    test_token = pd.read_csv("data/input/test_token.csv")
    test_token = test_token.dropna()
    print("Generating bert data")
    bert_test_data = generate_bert_data(test_token, tokenizer)
    bert_test_data['tag'] = "O"
    bert_test_data.loc[bert_test_data["ginza_bert_idx"] != 0, "tag"] = "mask"
    bert_test_data = bert_test_data[['token_id', 'tag', 'token', 'file_id', 'bert_tokens', 'bert_id', 'ginza_bert_idx']]
    print("Writing dict to file")
    file_ids = bert_test_data.file_id.unique()
    test_bert_data_dict = transfer_bert_data_to_dict(bert_test_data, file_ids, valid=True)
    path = f"data/preprocessed/test_bert_data_dict_{model_name}.pk"
    save_as_pickle(path, test_bert_data_dict)
    return test_bert_data_dict


def check(test_data):
    total = 0
    for d in test_data:
        l = d["labels"]
        total += len([i for i in l if i >= 0])
    assert total == 670092
