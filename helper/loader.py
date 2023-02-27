from typing import *
from helper import utils

import torch
import torch.utils.data
from transformers import AutoModel, AutoConfig, AutoTokenizer, BertModel, \
    BertTokenizer, ErnieForMaskedLM, AutoModelForMaskedLM
from openprompt.plms import load_plm
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader


def save_model(SAVE_PATH, epoch, model, optimizer):
    torch.save(
        obj={
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f=SAVE_PATH,
        _use_new_zipfile_serialization=False
    )


class PLMConfig:
    def __init__(self, mode):
        self.mode = mode
        self.plm = None
        self.tokenizer = None
        self.model_config = None
        self.WrapperClass = None
        self.pretrained = None
        self.promptTemplate = None

    def check(self):
        if self.mode == "pepl":
            return self.plm is not None and self.tokenizer is not None and self.model_config is not None \
                   and self.WrapperClass is not None
        elif self.mode == "base":
            return self.tokenizer is not None and self.pretrained
        else:
            return False


def choose_pretrained_model(name: str, load="pepl") -> PLMConfig:
    preconfig = PLMConfig(load)
    if load == "pepl":
        if name == "bert":
            preconfig.plm, preconfig.tokenizer, preconfig.model_config, preconfig.WrapperClass = \
                load_plm("bert", "bert-base-chinese")
        elif name == "roberta":
            preconfig.plm, preconfig.tokenizer, preconfig.model_config, preconfig.WrapperClass =\
                load_plm("bert", "hfl/chinese-roberta-wwm-ext")
        elif name == "ernie":
            standname = "nghuyong/ernie-3.0-base-zh"
            preconfig.plm = ErnieForMaskedLM.from_pretrained(standname)
            preconfig.tokenizer = BertTokenizer.from_pretrained(standname)
            preconfig.model_config = AutoConfig.from_pretrained(standname)
            preconfig.WrapperClass = MLMTokenizerWrapper
        elif name == "Erlangshen":
            standname = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"
            preconfig.tokenizer = AutoTokenizer.from_pretrained(standname, use_fast=False)
            preconfig.plm = AutoModelForMaskedLM.from_pretrained(standname)
            preconfig.model_config = AutoConfig.from_pretrained(standname)
            preconfig.WrapperClass = MLMTokenizerWrapper
        if preconfig.check():
            return preconfig
        else:
            raise RuntimeError('Please check the pretrained language model config')
    elif load == "base":
        if name == "bert":
            preconfig.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            preconfig.pretrained = AutoModel.from_pretrained('bert-base-chinese')
        elif name == "roberta":
            preconfig.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
            preconfig.pretrained = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        elif name == "ernie":
            preconfig.tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
            preconfig.pretrained = AutoModel.from_pretrained("nghuyong/ernie-3.0-base-zh")
        elif name == "Erlangshen":
            preconfig.tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese',
                                                                use_fast=False)
            preconfig.pretrained = AutoModel.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese')
        if preconfig.check():
            return preconfig
        else:
            raise RuntimeError('Please check the pretrained language model config')
    else:
        raise RuntimeError('Please configure proper return value of the pretrained model setting or check the load '
                           'argument')


def generate_dataloader(scale, few_shot, dataset, batch_size, plm_config, collate_fn, load="pepl"):
    def process_dataset(ori):
        ans = []
        for ite in ori:
            ans.append(InputExample(text_a=ite[0], label=ite[1]))
        return ans

    assert len(scale) == 2 or few_shot is not None, "Please check the argument for the script input."

    if few_shot is None:
        train_scale, test_scale = scale[0], scale[1]
        train_number, dev_number = int(len(dataset) * train_scale), len(dataset) - int(len(dataset) * test_scale)
        temp_train, temp_dev, temp_test = utils.split_dataset(dataset, train_number, dev_number)
    else:
        temp_train, temp_dev, temp_test = utils.few_split_dataset(dataset, few_shot)

    if load == "base":
        train_loader = torch.utils.data.DataLoader(dataset=temp_train, batch_size=batch_size, collate_fn=collate_fn,
                                                   shuffle=True, drop_last=True)
        dev_loader = torch.utils.data.DataLoader(dataset=temp_dev, batch_size=batch_size, collate_fn=collate_fn,
                                                 shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset=temp_test, batch_size=1, collate_fn=collate_fn,
                                                  shuffle=False, drop_last=False)
        return train_loader, dev_loader, test_loader

    elif load == "pepl":
        train_dataset, dev_dataset, test_dataset = process_dataset(temp_train), process_dataset(temp_dev), \
                                                   process_dataset(temp_test)

        train_loader = PromptDataLoader(
            dataset=train_dataset, tokenizer=plm_config.tokenizer, template=plm_config.promptTemplate,
            tokenizer_wrapper_class=plm_config.WrapperClass, batch_size=batch_size
        )
        dev_loader = PromptDataLoader(
            dataset=dev_dataset, tokenizer=plm_config.tokenizer, template=plm_config.promptTemplate,
            tokenizer_wrapper_class=plm_config.WrapperClass, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_title = torch.utils.data.DataLoader(
            dataset=temp_test, batch_size=1, collate_fn=collate_fn, shuffle=False, drop_last=False
        )
        test_loader = PromptDataLoader(
            dataset=test_dataset, tokenizer=plm_config.tokenizer, template=plm_config.promptTemplate,
            tokenizer_wrapper_class=plm_config.WrapperClass, batch_size=1, shuffle=False, drop_last=False
        )
        return train_loader, dev_loader, test_loader, test_title
