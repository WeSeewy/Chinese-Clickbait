import argparse
import os

import bert
import prompt
from helper import utils
from preprocess import ltp_tag_dataset
import torch
import warnings
warnings.filterwarnings("ignore")

TrainScale = 0.02       # The scale of train dataset
DevScale = 0.5          # The scale of test dataset
LearningRate = 5e-5     # The learning rate for AdamW
Epoch = 3               # Training epoch
RandomSeed = 42         # Random seed

DatasetPath = os.path.join("data", "all_labeled.csv")

if __name__ == '__main__':
    arg_description = "This script is an implement of PEPL(Part-of-speech Enhanced Prompt Learning) method for " \
                      "clickbait news detection. Our work has been accepted by [CSCWD'23]."
    parser = argparse.ArgumentParser(description=arg_description)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scale', '-s', nargs=2, type=float, help='Need 2 arguments to set the train and test scale of '
                                                                  'the dataset. The first input is for train and the '
                                                                  'second input is for test. For example, -s 0.01 0.5 '
                                                                  'means 1% of the dataset used for training, 50% of '
                                                                  'the dataset used for testing and others used '
                                                                  'for validation.')
    group.add_argument('--few_shot', '-f', type=int, help='Used for set few shot scenario. For example -f 16 means that'
                                                          'only 16 items in the dataset are used for training.')

    parser.add_argument('--batch_size', '-bs', type=int, default=8, help='the batch size used in training')
    parser.add_argument('--learning_rate', '-lr', type=float, default=LearningRate, help='learning rate for AdamW')
    parser.add_argument('--training_epoch', '-ep', type=int, default=Epoch, help='training epoch setting')
    parser.add_argument('--mode', '-m', type=str, default='pepl', choices=['pepl', 'base'],
                        help='the mode used for method, the pepl is default and the base is for baseline.')
    parser.add_argument('--pretrained_language_model', '-plm', type=str, default='bert',
                        choices=['bert', 'roberta', 'ernie', 'Erlangshen'],
                        help='the pretrained language model used for training and testing.')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.mode == 'pepl':
        DatasetPath = ltp_tag_dataset(DatasetPath)
        dataset = utils.wechat_only2_dataset(DatasetPath)
        prompt.run(device, RandomSeed, dataset, args.scale, args.few_shot, args.batch_size, args.learning_rate,
                   args.training_epoch, args.mode, args.pretrained_language_model)
    elif args.mode == 'base':
        dataset = utils.wechat_only2_dataset(DatasetPath)
        bert.run(device, RandomSeed, dataset, args.scale, args.few_shot, args.batch_size, args.learning_rate,
                 args.training_epoch, args.mode, args.pretrained_language_model)
    else:
        raise RuntimeError("Invalid mode, please check")