import torch
import numpy as np
class NERStackingDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, token_dict, labels=True):
        self.data_list = data_list
        self.labels = labels
        self.token_dict = token_dict
    def get_embedding(self, tokens):
        return np.stack([self.token_dict[token].cpu() for token in tokens])

    def __getitem__(self, idx):
        data = self.data_list[idx]
        item = {"input_logits": data["logits"],
                "embeddings": torch.tensor(self.get_embedding(data["tokens"]))
                }

        if self.labels:
            item['labels'] = torch.tensor(data["labels"])
        return item

    def __len__(self):
        return len(self.data_list)