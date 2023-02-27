import sys
sys.path.append("../")
from helper.metric import cal_marco_F1
from transformers import BertTokenizer
from helper.word import ltp_split

def unitest_cal_marco_F1(matrix):
    print(cal_marco_F1(matrix))


def unitest_bert_chinese(sent):
    print(sent)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print(tokenizer.encode(sent))


if __name__ == "__main__":
    matrix = [[454, 1879], [295, 4412]]
    # unitest_cal_marco_F1(matrix)
    sent = "10个词1个词2个词"
    print(unitest_bert_chinese(sent))