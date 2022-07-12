import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="")
parser.add_argument("--model_name", type=str)

parser.add_argument("--model_path", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
parser.add_argument("--seed", type=int, default=768)
parser.add_argument("--fold", type=int, default=5)

parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--valid_batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--warmup_ratio", type=float, default=0)
parser.add_argument("--toy", type=bool, default=False)
parser.add_argument("--all_data", type=bool, default=False)
# loss
parser.add_argument("--num_labels", type=int, default=11)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--valid_steps", type=float, default=100)


args = parser.parse_args()
for arg in ["lr", "weight_decay"]:
    args.__dict__[arg] = float(args.__dict__[arg])

for arg in vars(args):
    print(arg.rjust(20),":",getattr(args, arg))
