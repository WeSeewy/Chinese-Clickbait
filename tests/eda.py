import os, sys
sys.path.append("../")
from helper import loader
from typing import *


def label_count(dataset):
    ans = [0 for _ in range(3)]
    for ite in dataset:
        label = int(ite["label"])
        ans[label] += 1
    return ans


def distinct_dataset(dataset):
    text_set = set()
    ans_dataset = []
    for ite in dataset:
        if ite["text"] not in text_set:
            text_set.add(ite["text"])
            ans_dataset.append(ite)
    return ans_dataset


if __name__ == "__main__":
    wechat_path = os.path.join("..", os.path.join("data", "all_labeled.csv"))
    wechat_dataset = loader.wechat_only2_dataset(wechat_path)
    label_lis = label_count(wechat_dataset)
    print(label_lis)
    d_dataset = distinct_dataset(wechat_dataset)
    print(label_count(d_dataset))

