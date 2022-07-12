import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
from utils import id2tag, save_model, save
import torchcontrib


class MAMeter:
    def __init__(self, windows=10):
        self.windows = windows
        self.ls = []

    def add(self, num):
        if len(self.ls) < 10:
            self.ls = self.ls + [num]
        else:
            self.ls = self.ls[1:] + [num]

    def avg(self):
        return sum(self.ls) / len(self.ls)


loss_fct = CrossEntropyLoss()


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    loss_meter = MAMeter(10)
    bar = tqdm(data_loader, total=len(data_loader))

    for _, batch in enumerate(bar):
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        labels = batch["labels"].to(device, dtype=torch.long)
        attention_mask = input_ids != 0

        model.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.add(loss.item())
        bar.set_postfix(loss=loss_meter.avg())


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
            tags, labels, token_logits = get_result(labels_ids, logits,return_logits=True)
            logits_list.append(token_logits.cpu())
            tag_list.extend(tags)
            label_list.extend(labels)
    logits_list = torch.cat(logits_list)
    report = classification_report([label_list], [tag_list], digits=4)
    return tag_list, label_list, logits_list, report


def train_with_valid(train_dataloader,
                     valid_dataloader,
                     test_dataloader,
                     model, optimizer,
                     device,
                     scheduler,
                     valid_steps,
                     save_dir,
                     output_dir,
                     model_dir,
                     fold,
                     wandb
                     ):
    model.train()
    loss_meter = MAMeter(10)
    bar = tqdm(train_dataloader, total=len(train_dataloader))

    for step, batch in enumerate(bar):
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        labels = batch["labels"].to(device, dtype=torch.long)
        attention_mask = input_ids != 0

        model.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_meter.add(loss.item())
        bar.set_postfix(loss=loss_meter.avg())
        wandb.log({"loss": loss.item()})
        if (step > 0 and step % valid_steps == 0) or step == len(train_dataloader) - 1:
            print("##########################################")
            print(f"Start evaluation {step}")
            tag_list, label_list, report = valid_fn(valid_dataloader, model, device)
            with open(f"{save_dir}/classify_report_{fold}_{step}.txt", "w") as f:
                f.write(report)
            save([tag_list, label_list, report], f"{output_dir}/valid_pred_{fold}_{step}.pk")
            report = classification_report([label_list], [tag_list], digits=4,output_dict=True)
            for key, item in report.items():
                if isinstance(item, dict):
                    for sub_key, sub_item in item.items():
                        wandb.log({f"{key}_{sub_key}": sub_item})

                else:
                    wandb.log({key: item})

    test_tags, _, _ = valid_fn(test_dataloader, model, device)
    save(test_tags, f"{output_dir}/test_pred_{fold}.pk")
    optimizer.swap_swa_sgd()
    save_model(model, f"{model_dir}/model_{fold}.pt")
