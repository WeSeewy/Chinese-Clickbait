import os
import pandas as pd

from ltp import LTP


def ltp_split(ltp, sent):
    output = ltp.pipeline(sent, tasks=["cws", "pos"])
    word_lis = ["[SEP]"]
    o_cws = output.cws
    o_pos = output.pos
    chosen_name = ['d', 'm', 'r', 'wp', 'o', 'nh', 'ni', 'ns']

    for i in range(len(o_cws)):
        prop = o_pos[i]
        if prop in chosen_name:
            word_lis.append(o_cws[i])
    return sent + str("".join(word_lis))


def ltp_tag_dataset(path):
    ltp = LTP("LTP/base1")
    pd_dataframe = pd.read_csv(path)
    dataset_texts = pd_dataframe["TITLE"].values.tolist()
    new_texts = []
    dataset_number = len(dataset_texts)
    print("Start PoS tagging dataset...")
    finished_10per = dataset_number // 10
    cnt = 0
    for i in range(dataset_number):
        if i % finished_10per == 0:
            print("Finished: " + str(cnt) + "%")
            cnt += 10
        new_item = ltp_split(ltp, dataset_texts[i])
        new_texts.append(new_item)
    file_path = os.path.join("data", "ltp_labeled.csv")
    pd_dataframe["TITLE"] = new_texts
    pd_dataframe.to_csv(file_path)
    print("Finished Processing Dataset")
    return file_path


if __name__ == "__main__":
    ltp_tag_dataset(os.path.join("data", "all_labeled.csv"))