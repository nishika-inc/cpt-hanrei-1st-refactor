import torch
from tqdm import tqdm
from seqeval.metrics import classification_report

from dataloader import create_dataloader
from utils import model_path_map,  id2tag, save_as_pickle
from model import NERModel


def get_result(labels_ids, logits, return_logits=False):
    labels_ids = labels_ids.flatten()
    if return_logits:
        token_logits = logits.view(-1, 11)
        token_logits = token_logits[labels_ids >= 0, :]
    tag_ids = logits.argmax(axis=-1).flatten()
    tag_ids = tag_ids[labels_ids >= 0].tolist()
    labels_ids = labels_ids[labels_ids >= 0].tolist()
    tags = [id2tag[id] for id in tag_ids]
    labels = [id2tag[id] for id in labels_ids]
    if return_logits:
        return tags, labels, token_logits
    return tags, labels


def valid_fn(data_loader, model, device):
    model.eval()
    tag_list = []
    label_list = []
    logits_list = []
    with torch.no_grad():
        bar = tqdm(data_loader, total=len(data_loader))
        for _, batch in enumerate(bar):
            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            labels_ids = batch["labels"].to(device, dtype=torch.long)
            attention_mask = input_ids != 0
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            tags, labels, token_logits = get_result(
                labels_ids, logits, return_logits=True)
            logits_list.append(token_logits.cpu())
            tag_list.extend(tags)
            label_list.extend(labels)
    logits_list = torch.cat(logits_list)
    report = classification_report([label_list], [tag_list], digits=4)
    return tag_list, label_list, logits_list, report


if __name__ == '__main__':
    trail_name = "seed_data_for_each_model"
    valid_batch_size = 64

    for model_name in model_path_map.keys():
        print(model_name)
        model = NERModel.from_pretrained(model_path_map[model_name],
                                         output_hidden_states=True,
                                         num_labels=11
                                         )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_path = f"../data/preprocessed/test_bert_data_dict_{model_name}.pk"
        test_dataloader = create_dataloader(test_path, valid_batch_size)
        for fold in range(5):
            model_path = f"../save/train/{trail_name}_{model_name}/model/model_{fold}.pt"
            model.load_state_dict(torch.load(model_path))
            test_tags, test_labels, logits_list, report = valid_fn(
                test_dataloader, model, device)
            save_path = f"../save/train/{trail_name}_{model_name}/output/test_pred_{fold}.pk"
            save_as_pickle(
                save_path, [test_tags, test_labels, logits_list, report])
