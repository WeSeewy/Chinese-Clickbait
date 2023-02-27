import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def wechat_only2_dataset(path: str) -> list:
    pd_dataframe = pd.read_csv(path)
    dataset_texts = pd_dataframe["TITLE"].values.tolist()
    dataset_labels = [int(x) for x in pd_dataframe["T_LABEL"].values.tolist()]
    assert len(dataset_texts) == len(dataset_labels)
    dataset = []
    for i in range(len(dataset_texts)):
        if dataset_labels[i] == 0:
            dataset.append({"text": dataset_texts[i], "label": dataset_labels[i]})
        elif dataset_labels[i] == 1:
            dataset.append({"text": dataset_texts[i], "label": 1})
        elif dataset_labels[i] == 2:
            dataset.append({"text": dataset_texts[i], "label": 1})
    return dataset


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # [{"text": ..., "label": ...}]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]["text"]
        label = self.dataset[index]["label"]

        return text, label


def split_dataset(dataset: list, train_index: int, dev_index:int):
    train_dataset = CustomDataset(dataset[:train_index])
    dev_dataset = CustomDataset(dataset[train_index:dev_index])
    eval_dataset = CustomDataset(dataset[dev_index:])
    return train_dataset, dev_dataset, eval_dataset


# few-shot training split
def few_split_dataset(dataset: list, shot: int) -> (CustomDataset, CustomDataset, CustomDataset):
    train_dataset, eval_dataset = [], []
    label_num = [0, 0]
    item_num = len(dataset)
    search_num = 0
    shot_lis = [shot, shot]
    for i in range(item_num):
        if label_num[0] >= shot_lis[0] and label_num[1] >= shot_lis[1]:
            search_num = i
            break
        cur_item = dataset[i]
        cur_label = int(cur_item["label"])
        if label_num[cur_label] < shot_lis[cur_label]:
            train_dataset.append(cur_item)
            label_num[cur_label] += 1
        else:
            eval_dataset.append(cur_item)
    assert search_num != 0
    print(search_num)
    rest_index = (item_num - 1 - search_num) // 2
    dev_dataset = dataset[search_num: search_num + rest_index]
    eval_dataset.extend(dataset[search_num + rest_index:])
    print("the len of train_dataset: " + str(len(train_dataset)))
    print("the len of dev_dataset: " + str(len(dev_dataset)))
    print("the len of eval_dataset: " + str(len(eval_dataset)))
    return CustomDataset(train_dataset), CustomDataset(dev_dataset), CustomDataset(eval_dataset)


def set_collate_fn(tokenizer, device):

    def collate_fn(data):
        sents = [i[0] for i in data]
        labels = [i[1] for i in data]

        encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                             truncation=True,
                                             padding='max_length',
                                             max_length=256,
                                             return_tensors='pt',
                                             return_length=True)

        input_ids = torch.tensor(encode["input_ids"], device=device)
        attention_mask = torch.tensor(encode["attention_mask"], device=device)
        token_type_ids = torch.tensor(encode["token_type_ids"], device=device)
        labels = torch.tensor(labels, device=device)
        return sents, input_ids, attention_mask, token_type_ids, labels

    return collate_fn