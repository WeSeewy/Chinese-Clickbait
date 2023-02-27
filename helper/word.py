import torch
from ltp import LTP


def ltp_split(ltp, sent_list):
    output_lis = ltp.pipeline(sent_list, tasks=["cws", "pos"])
    o_cws_lis = output_lis.cws
    o_pos_lis = output_lis.pos
    ans_lis = []
    for i in range(len(o_cws_lis)):
        o_cws = o_cws_lis[i]
        word_lis = ["[SEP]"]
        for j in range(len(o_cws)):
            prop = o_pos_lis[i][j]
            if prop == 'd' or prop == 'm' or prop == 'r' or prop == 'wp':
                word_lis.append(o_cws[j])
        ans_lis.append(sent_list[i] + str("".join(word_lis)))
    return ans_lis


if __name__ == "__main__":
    ltp = LTP("LTP/base1")
    output = ltp_split(ltp, ["不懂这些互联网黑话，都没法聊天", "测试这句话", "数据集构建"])
    print(output)