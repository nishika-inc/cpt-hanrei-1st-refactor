import os
import numpy as np
import random

from seqeval.metrics import classification_report

from layer1.args import args
import torch
from transformers import get_linear_schedule_with_warmup, TrainingArguments, Trainer
from layer1.dataloader import create_dataloader
from layer1.loops import valid_fn
from layer1.model import get_optimizer, NERModel
from layer1.utils import save, get_save_dir, model_path_map,seed_map
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

#
valid_path = f"data/preprocessed/valid_bert_data_dict_orig_{args.model_name}_seed{seed}_{args.fold}.pk"
test_path = f"data/preprocessed/test_bert_data_dict_{args.model_name}.pk"

train_data_list = args.train_data_list.split(",")
train_dataloader = create_dataloader(train_data_list,
                                     args.train_batch_size,
                                     shuffle=True,
                                     seed=args.seed)
valid_dataloader = create_dataloader(valid_path, args.valid_batch_size)
test_dataloader = create_dataloader(test_path, args.valid_batch_size)
print(str(args.fold) * 50)
print("Reading model...")
# model and optimizer

# model = NERModel.from_pretrained(args.model_path,
#                                  output_hidden_states=True,
#                                  num_labels=args.num_labels
#                                  )

model_path = model_path_map[args.model_name]
model = NERModel.from_pretrained(model_path,
                                 output_hidden_states=True,
                                 num_labels=args.num_labels
                                 )
num_train_steps = int(len(train_dataloader.dataset) / args.train_batch_size * args.epochs)
optimizer = get_optimizer(model, args.lr, args.weight_decay)

# optimizer = torchcontrib.optim.SWA(
#     base_opt,
#     swa_start=100,
#     swa_freq=100,
#     swa_lr=None)
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
    save_steps=10000,
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
print("saving")
with open(f"{save_dir}/classify_report_{args.fold}.txt", "w") as f:
    f.write(report)
save([tag_list, label_list, logits_list, report], f"{output_dir}/valid_pred_{args.fold}.pk")
report = classification_report([label_list], [tag_list], digits=4)
print(report)


