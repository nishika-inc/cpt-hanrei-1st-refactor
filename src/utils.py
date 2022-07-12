import os
import pickle
from shutil import copyfile

import torch
from seqeval.metrics import f1_score, classification_report

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

model_path_map = {
    "cl-wom": "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-charwom": "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "cl": "cl-tohoku/bert-base-japanese",
    "cl-char": "cl-tohoku/bert-base-japanese-char",
    "NICT-100k":"data/model/NICT_BERT-base_JapaneseWikipedia_100K",
    "NICT-32k":"data/model/NICT_BERT-base_JapaneseWikipedia_32K_BPE"
}

seed_map = {
    "cl-wom": 71,
    "cl-charwom": 271,
    "cl": 4306,
    "cl-char": 1545,
    "NICT-100k": 8155,
    "NICT-32k": 1250,
}

def compute_metrics(pred):
    labels_ids = pred.label_ids.flatten()
    tag_ids = pred.predictions.flatten()
    tag_ids = tag_ids[labels_ids >= 0].tolist()
    labels_ids = labels_ids[labels_ids >= 0].tolist()
    tags = [id2tag[id] for id in tag_ids]
    labels = [id2tag[id] for id in labels_ids]
    f1 = f1_score([labels], [tags])
    return {"f1": f1}


def save(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)

def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    subdir = 'train' if training else 'test'
    save_dir = os.path.join(base_dir, subdir, f'{name}')
    output_dir = os.path.join(save_dir, f'output')
    model_dir = os.path.join(save_dir, f'model')
    for dir in [save_dir, output_dir, model_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    save_py_files(save_dir)
    return save_dir, output_dir, model_dir

def save_py_files(save_dir):

    files = [f for f in os.listdir('.') if str(f).endswith(".py")]
    for f in files:
        copyfile(f, os.path.join(save_dir, f))

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)