import os
import numpy as np
import random

from seqeval.metrics import classification_report

from args import args
import torch
from transformers import get_linear_schedule_with_warmup, BertForTokenClassification, TrainingArguments, Trainer
from dataloader import create_dataloader, get_toy_dataloader, get_all_dataloader
from loops import train_fn, valid_fn, train_with_valid
from model import get_optimizer, NERModel
from utils import save, get_save_dir, save_model, model_path_map, seed_map
import wandb

print("changed")
wandb.init(config=args)

save_dir, output_dir, model_dir = get_save_dir("save/", f"{args.name}_{args.model_name}", training=not args.toy)
run_name = save_dir.split("/")[-1]
print(run_name)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed = seed_map[args.model_name]
seed_everything(args.seed)
print("start read data")
train_path1 = f"data/preprocessed/train_bert_data_dict_orig_{args.model_name}_seed{seed}_{args.fold}.pk"
train_path2 = f"data/preprocessed/train_bert_data_dict_aug_0_{args.model_name}_seed{seed}_{args.fold}.pk"
train_path3 = f"data/preprocessed/train_bert_data_dict_aug_1_{args.model_name}_seed{seed}_{args.fold}.pk"
train_path4 = f"data/preprocessed/train_bert_data_dict_aug_2_{args.model_name}_seed{seed}_{args.fold}.pk"

valid_path = f"data/preprocessed/valid_bert_data_dict_orig_{args.model_name}_seed{seed}_{args.fold}.pk"
test_path = f"data/preprocessed/test_bert_data_dict_{args.model_name}.pk"

train_dataloader = create_dataloader([train_path1, train_path2, train_path3
                                      # valid_path, valid_path2, valid_path3
                                      ],
                                     args.train_batch_size,
                                     shuffle=True,
                                     seed=args.seed)
valid_dataloader = create_dataloader(valid_path, args.valid_batch_size)
test_dataloader = create_dataloader(test_path, args.valid_batch_size)
print(str(args.fold) * 50)
print("Reading model...")


model_path = model_path_map[args.model_name]
model = NERModel.from_pretrained(model_path,
                                 output_hidden_states=True,
                                 num_labels=args.num_labels
                                 )
num_train_steps = int(len(train_dataloader.dataset) / args.train_batch_size * args.epochs)
optimizer = get_optimizer(model, args.lr, args.weight_decay)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(num_train_steps * args.warmup_ratio),
    num_training_steps=num_train_steps
)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.valid_batch_size,
    warmup_steps=100,
    logging_dir='./logs',
    learning_rate=args.lr,
    seed=args.seed,
    save_steps=100000,
    do_eval=False,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader.dataset,
    optimizers=(optimizer, scheduler)
)
trainer.train()

print("##########################################")
print(f"Start evaluation")
tag_list, label_list, logits_list, report = valid_fn(valid_dataloader, model, device)
with open(f"{save_dir}/classify_report_{args.fold}.txt", "w") as f:
    f.write(report)
save([tag_list, label_list, logits_list, report], f"{output_dir}/valid_pred_{args.fold}.pk")
report = classification_report([label_list], [tag_list], digits=4)
print(report)
print("saving")
test_tags, test_labels, logits_list, report = valid_fn(test_dataloader, model, device)
save([test_tags,test_labels, logits_list, report], f"{output_dir}/test_pred_{args.fold}.pk")
save_model(model, f"{model_dir}/model_{args.fold}.pt")