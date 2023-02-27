import os
import random
from helper import loader, utils, metric

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data


writer = SummaryWriter()


class DownModel(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)
        self.pretrained = pretrained

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


def test(args_mode, args_plm, model, test_loader):
    model.eval()
    correct, total = 0, 0
    types_number = 2
    matrix = [[0 for _ in range(types_number)] for _ in range(types_number)]
    result_save_path = os.path.join("results", str(args_mode) + "_" + str(args_plm) + "_case_res.txt")

    with open(result_save_path, "w", encoding="utf-8") as case_f:
        for i, (title, input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
            with torch.no_grad():
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            case_f.writelines([str(labels[0].tolist()), "\t", str(out[0].tolist()), "\t", title[0], "\n"])

            out = out.argmax(dim=1)
            cor, tot = metric.accuracy(out, labels)
            correct += cor
            total += tot
            cur_mat = metric.multi_label_metric(out, labels, types_number)
            for j in range(types_number):
                for k in range(types_number):
                    matrix[j][k] += cur_mat[j][k]

    print("Accuracy: " + str(correct / total))
    print("Confusion Matrix")
    for i in range(types_number):
        cur = []
        for j in range(types_number):
            cur.append(str(matrix[i][j]))
        print(" ".join(cur))
    print("marco_Precision: "+str(metric.cal_marco_Pre(matrix)))
    print("marco_Recall: "+str(metric.cal_marco_Rec(matrix)))
    print("marco_F1: " + str(metric.cal_marco_F1(matrix)))
    return correct / total


def run(device, seed, dataset, args_scale, args_fewshot, args_batch_size, args_learning_rate, args_training_epoch,
        args_mode, args_plm):

    utils.setup_seed(seed)

    plm_config = loader.choose_pretrained_model(args_plm, load=args_mode)
    tokenizer, pretrained = plm_config.tokenizer, plm_config.pretrained

    pretrained.to(device)
    random.shuffle(dataset)
    collate_fn = utils.set_collate_fn(tokenizer, device)

    train_loader, dev_loader, test_loader = loader.generate_dataloader(args_scale, args_fewshot, dataset,
                                                                       args_batch_size, plm_config, collate_fn,
                                                                       load="base")
    model = DownModel(plm_config.pretrained).to(device)

    optimizer = AdamW(model.parameters(), lr=args_learning_rate)

    weights = [2.0, 1.0]
    class_weights = torch.FloatTensor(weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(device)

    epoch = args_training_epoch  # recommend 2-4
    print("Start Training: ")
    print("[Mode]: "+str(args_mode))
    print("[Pretrained Language Model]: "+str(args_plm))
    for exec_index in range(epoch):
        model.train()
        loss_cum = 0
        for i, (_, input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_cum += loss.item()
            if i % 5 == 0:
                print(i, loss_cum / (i + 1))
                writer.add_scalar(str(exec_index) + " Exec Loss", loss_cum / (i + 1), i)
        model_save_path = os.path.join("checkpoints", str(args_mode)+"_"+str(args_plm)+"_fc_"+str(exec_index)+".pt")
        loader.save_model(model_save_path, exec_index, model, optimizer)

    test(args_mode, args_plm, model, test_loader)

