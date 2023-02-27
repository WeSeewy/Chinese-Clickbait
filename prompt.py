import os
from helper import utils, loader, metric

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification

writer = SummaryWriter()


def run(device, seed, dataset, args_scale, args_fewshot, args_batch_size, args_learning_rate, args_training_epoch,
        args_mode, args_plm):

    utils.setup_seed(seed)

    # There are two classes in Clickbait Detection, one for normal and one for clickbait
    classes = [
        "normal",
        "clickbait"
    ]

    plm_config = loader.choose_pretrained_model(args_plm, load=args_mode)

    plm, tokenizer, model_config, WrapperClass = plm_config.plm, plm_config.tokenizer, \
                                                 plm_config.model_config, plm_config.WrapperClass

    collate_fn = utils.set_collate_fn(tokenizer, device)

    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} 这是 {"mask"} 标题',
        tokenizer=plm_config.tokenizer
    )

    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "normal": ["准确", "合适", "合规", "规范", "合格", "高质量"],
            "clickbait": ["标题党", "误导", "诱导", "诱饵", "暗示", "失实"],
        },
        tokenizer=tokenizer,
    )

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    plm_config.promptTemplate = promptTemplate
    promptModel.to(device)
    print(promptModel.template.text)

    train_loader, dev_loader, test_loader, title_loader = loader.generate_dataloader(
        args_scale, args_fewshot, dataset, args_batch_size, plm_config, collate_fn, load="pepl"
    )

    optimizer = AdamW(promptModel.parameters(), lr=args_learning_rate)

    weights = [2.0, 1.0]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(device)

    epoch = args_training_epoch  # recommend 2-4
    for exec_index in range(epoch):
        promptModel.train()
        loss_cum = 0
        ind = 0
        for batch in train_loader:
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            logits = promptModel(batch)
            labels = batch["label"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_cum += loss.item()
            if ind % 5 == 0:
                print(ind, loss_cum / (ind + 1))
                writer.add_scalar(str(exec_index) + " Exec Loss", loss_cum/(ind+1), ind)
            ind += 1
        loader.save_model(os.path.join("checkpoints", "bert_prompt_" + str(exec_index) + ".pt"), exec_index,
                          promptModel, optimizer)

    test(device, args_mode, args_plm, promptModel, test_loader, title_loader)


def test(device, args_mode, args_plm, promptModel, test_loader, title_loader):
    promptModel.eval()
    correct, total = 0, 0
    types_number = 2
    matrix = [[0 for _ in range(types_number)] for _ in range(types_number)]

    cnt = 0
    ans_list = []
    for _, (title, input_ids, attention_mask, token_type_ids, labels) in enumerate(title_loader):
        ans_list.append([title, labels])
    with torch.no_grad():
        result_save_path = os.path.join("results", str(args_mode) + "_" + str(args_plm) + "_case_res.txt")
        with open(result_save_path, "w", encoding="utf-8") as case_f:
            for batch in test_loader:
                title, th_label =ans_list[cnt]
                for k in batch.keys():
                    batch[k] = batch[k].to(device)
                logits = promptModel(batch)
                preds = torch.argmax(logits, dim=-1)

                labels = batch["label"]
                if th_label != labels[0].tolist():
                    print("error")
                case_f.writelines([str(labels[0].tolist()), "\t", str(logits[0].tolist()), "\t", title[0], "\n"])

                cor, tot = metric.accuracy(preds, labels)
                correct += cor
                total += tot
                cur_mat = metric.multi_label_metric(preds, labels, types_number)
                for j in range(types_number):
                    for k in range(types_number):
                        matrix[j][k] += cur_mat[j][k]
                cnt += 1

    print("Accuracy: " + str(correct / total))
    print("Confusion Matrix")
    for i in range(types_number):
        cur = []
        for j in range(types_number):
            cur.append(str(matrix[i][j]))
        print(" ".join(cur))
    print("marco_Precision: " + str(metric.cal_marco_Pre(matrix)))
    print("marco_Recall: " + str(metric.cal_marco_Rec(matrix)))
    print("marco_F1: " + str(metric.cal_marco_F1(matrix)))
    return correct / total
