import pickle
import random
import torch
from torch.utils.data import DataLoader


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=True):
        self.data = data
        self.labels=labels

    def __getitem__(self, idx):
        data = self.data[idx]
        item = {"input_ids": torch.tensor(data["ids"]),
                # "mask": torch.tensor(data["mask"])
        }
        if self.labels:
            item['labels'] = torch.tensor(data["labels"])
        return item

    def __len__(self):
        return len(self.data)

def create_dataloader(path, batch_size,shuffle=False, seed=0):
    if isinstance(path,list):
        bert_data_dict = []
        for p in path:
            with open(p, 'rb') as web:

                data_list = pickle.load(web)
                if shuffle:
                    random.Random(seed).shuffle(data_list)
                bert_data_dict.extend(data_list)
    else:
        with open(path, 'rb') as web:
            bert_data_dict = pickle.load(web)

    dataset = NERDataset(bert_data_dict)
    return DataLoader(dataset, batch_size=batch_size)

def get_toy_dataloader():
    path = f"data/preprocessed/train_bert_data_dict_0.pk"
    with open(path, 'rb') as web:
        bert_data_dict = pickle.load(web)
    dataset = NERDataset(bert_data_dict[:10])
    return DataLoader(dataset, batch_size=2)

def get_all_dataloader(train_path, valid_path, batch_size, shuffle=False):
    with open(train_path, 'rb') as web:
        trian_data_dict = pickle.load(web)
    with open(valid_path, 'rb') as web:
        valid_data_dict = pickle.load(web)
    dataset = NERDataset(trian_data_dict+valid_data_dict)
    tinyset = NERDataset(trian_data_dict[:10])
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    valid_data = DataLoader(tinyset, batch_size=batch_size, shuffle=shuffle)
    return train_data, valid_data
