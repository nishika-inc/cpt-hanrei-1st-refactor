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


