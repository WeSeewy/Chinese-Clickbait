from typing import *


def accuracy(out, labels) -> tuple:
    return (out == labels).sum().item(), len(labels)


def multi_label_metric(out, labels, types_number) -> list[list]:
    confusion_matrix = [[0 for _ in range(types_number)] for _ in range(types_number)]
    for i in range(len(out)):
        confusion_matrix[labels[i]][out[i]] += 1
    return confusion_matrix


def cal_marco_Pre(conf_matrix):
    dim = len(conf_matrix)
    assert dim == 2
    p0 = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
    p1 = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
    return (p0 + p1) / 2


def cal_marco_Rec(conf_matrix):
    dim = len(conf_matrix)
    assert dim == 2
    r0 = conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[0][1])
    r1 = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    return (r0 + r1) / 2


def cal_marco_F1(conf_matrix):
    dim = len(conf_matrix)
    assert dim == 2
    p0 = conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[1][0])
    r0 = conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[0][1])
    f1_0 = 2 * (p0 * r0) / (p0 + r0)

    p1 = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
    r1 = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    f1_1 = 2 * (p1 * r1) / (p1 + r1)

    return (f1_0 + f1_1)/2
