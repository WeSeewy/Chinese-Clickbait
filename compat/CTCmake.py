# ref: https://github.com/649453932/Chinese-Text-Classification-Pytorch

import os, sys
sys.path.append("../")
from helper import loader, utils
import random


if __name__ == "__main__":
    scal = 0.6
    utils.setup_seed(42)
    path = os.path.join(os.path.join(os.path.join(".."), "data"), "all_labeled.csv")
    wechat_dataset = utils.wechat_only2_dataset(path)

    random.shuffle(wechat_dataset)
    train_number = int(len(wechat_dataset) * scal)
    dev_number = int(len(wechat_dataset) * 0.8)

    train_dataset, dev_dataset, eval_dataset = utils.split_dataset(wechat_dataset, train_number, dev_number)
    # train_dataset, dev_dataset, eval_dataset= utils.few_split_dataset(wechat_dataset, 32)
    all_data = [train_dataset, dev_dataset, eval_dataset]
    all_name = ["train.txt", "dev.txt", "test.txt"]

    for i in range(len(all_data)):
        this_dataset = all_data[i]
        this_path = os.path.join("data", all_name[i])
        ans_lis = []
        with open(this_path, "w", encoding="utf-8") as w_f:
            for dic in this_dataset:
                ans_lis.append(str(dic[0]+"\t"+str(dic[1])+"\n"))
            w_f.writelines(ans_lis)
    print("finished: "+str(scal))
