import torch
from utils import id2tag, load_pickle
from tqdm import tqdm
from model import NERStackingModel
from dataloader import NERStackingDataset
from torch.utils.data import DataLoader
from post_processing import correct_idx, get_voted_result
import pandas as pd


def test_fn(data_loader, model, device):
    model.eval()
    model.to(device)
    logits_list = []
    with torch.no_grad():
        bar = tqdm(data_loader, total=len(data_loader))
        for _, batch in enumerate(bar):
            input_logits = batch["input_logits"].to(device)
            embeddings = batch["embeddings"].to(device)
            outputs = model(input_logits=input_logits, embeddings=embeddings)
            logits = outputs[0]
            logits_list.append(logits)
    pred = torch.cat(logits_list, axis=1).argmax(-1).detach().cpu().numpy()
    return [id2tag[id] for id in pred[0]]


def save_result(tags, file_name):
    sub = pd.read_csv("../data/input/sample_submission.csv")
    tokens = pd.read_csv("../data/input/test_token.csv")
    tokens = tokens.dropna()
    tokens["tag"] = tags
    tokens = tokens.sort_values(["file_id", "token_id"])
    submission = sub.drop("tag", axis=1).merge(
        tokens[["file_id", "token_id", "tag"]], how="outer")
    submission.to_csv(file_name, index=False)


if __name__ == '__main__':
    tag_list = []
    hidden_dim = 64
    model_num = 6
    token_dict = load_pickle("../data/preprocessed/flair_embedding_dict.pk")
    stacking_model = NERStackingModel(hidden_dim, model_num)
    stacking_model.load_state_dict(torch.load(
        f"../save/stacking/stacking_model_2.pt"))
    # for fold in range(5):
    #     test_path = f"../data/preprocessed/test_stacking_data_{fold}.pk"
    #     test_data = load_pickle(test_path)
    #     test_dataset = NERStackingDataset(test_data, token_dict, labels=False)
    #     test_loader = DataLoader(test_dataset, batch_size=1)
    #     tags = test_fn(test_loader, stacking_model, "cuda")
    #     tag_list.append(tags)
    #
    #
    #
    #
    tokens = pd.read_csv("../data/input/test_token.csv").dropna().token.tolist()
    # corrected_list = [correct_idx(t, tokens) for t in tag_list]
    # voted = get_voted_result(corrected_list)
    # corrected = correct_idx(voted, tokens)
    #
    # save_result(corrected, f"stacking_vote.csv")

    test_path = f"../data/preprocessed/test_stacking_data_mean.pk"
    test_data = load_pickle(test_path)
    test_dataset = NERStackingDataset(test_data, token_dict, labels=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    tags = test_fn(test_loader, stacking_model, "cuda")
    # "B-"がなく"I-"から始まっているタグを"O"にする後処理
    corrected = correct_idx(tags, tokens)
    save_result(corrected, f"../data/output/stacking_infer_only.csv")
