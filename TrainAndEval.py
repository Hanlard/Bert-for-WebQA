import os
import json
from torch.utils import data
from transformers import BertTokenizer , BertModel
import numpy as np
import argparse

import torch
import torch.nn as nn
from model import Net
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
# 字符ID化

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class WebQADataset(data.Dataset):
    def __init__(self, fpath):

        self.questions, self.evidences, self.answer= [], [], []
        with open(fpath, 'r',encoding='utf-8') as f:
            data = json.load(f)#读取json文件内容
            for key in data:
                item = data[key]
                question = item['question']
                evidences = item['evidences']
                for evi_key in evidences:
                    evi_item = evidences[evi_key]
                    self.questions.append(question)
                    self.evidences.append(evi_item['evidence'])
                    self.answer.append(evi_item['answer'][0])

    def __len__(self):
        return len(self.answer)

    def FindOffset(self, tokens_id, answer_id):
        n = len(tokens_id)
        m = len(answer_id)
        if n < m:
            return False
        for i in range(n - m + 1):
            if tokens_id[i:i + m] == answer_id:
                return (i, i + m)
        return False

    def __getitem__(self, idx):
        # We give credits only to the first piece.
        q, e, a = self.questions[idx], self.evidences[idx], self.answer[idx]
        '''
        sen = "我是中国人，你呢？"
        tokens = tokenizer.tokenize(sen)
        tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
        print(tokens)
        print(tokens_xx)
        ['我', '是', '中', '国', '人', '，', '你', '呢', '？']
        [2769, 3221, 704, 1744, 782, 8024, 872, 1450, 8043]
        '''
        ## 测试集中最长 Evidence 244，最长 Answer 89，所以将 max_len = 400
        tokens = tokenizer.tokenize('[CLS]'+q+'[SEP]'+e)# list
        if len(tokens)>512:
            tokens=tokens[:512]
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        answer_offset = (-1, -1)
        answer_seq_label = len(tokens_id) * [0]
        if a != 'no_answer':
            answer_tokens = tokenizer.tokenize(a)
            answer_offset = self.FindOffset(tokens, answer_tokens)#有肯能返回False
            if answer_offset:#在原文中找到答案
                answer_seq_label[answer_offset[0]:answer_offset[1]] = [1]*(len(answer_tokens))
            else:# self.FindOffset 返回False
                answer_offset = (-1, -1)


        return tokens, tokens_id, answer_offset, answer_seq_label

    def get_samples_weight(self):
        samples_weight = []
        for i in range(len(self.answer)):
            tokens ,tokens_id, answer_offset, answer_seq_label = self.__getitem__(i)
            if answer_offset != (-1,-1):
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)

def pad(batch):
    tokens_l, tokens_id_l, answer_offset_l, answer_seq_label_l= list(map(list, zip(*batch)))
    maxlen = np.array([len(sen) for sen in tokens_l]).max()
    ### pad和截断
    for i in range(len(tokens_l)):
        tokens = tokens_l[i]
        tokens_id= tokens_id_l[i]
        answer_offset = answer_offset_l[i]
        answer_seq_label = answer_seq_label_l[i]
        tokens_l[i] = tokens + (maxlen - len(tokens))*['[PAD]']
        tokens_id_l[i] =tokens_id + (maxlen - len(tokens))*tokenizer.convert_tokens_to_ids(['[PAD]'])
        answer_seq_label_l[i] = answer_seq_label + [0]*(maxlen - len(tokens))
    return tokens_l, tokens_id_l, answer_offset_l, answer_seq_label_l

def result_metric(prediction_all, y_2d_all):
    total_num=0
    toral_cur=0
    for prediction, y_2d in zip(prediction_all, y_2d_all):
        batch_size,seq_len = prediction.size()
        currect = torch.sum(torch.sum(prediction == y_2d, dim=1)==seq_len).to("cpu").item()
        toral_cur = toral_cur + currect
        total_num = total_num + batch_size
    return toral_cur/total_num

def TrainOneEpoch(model, train_iter, optimizer, hp):
    model.train()
    for i, batch in enumerate(train_iter):
        _, tokens_id_l, answer_offset_l, answer_seq_label_l = batch
        optimizer.zero_grad()
        prediction, loss, y_2d  = model.module.forward(tokens_id_l, answer_offset_l, answer_seq_label_l)
        # nn.utils.clip_grad_norm_(model.parameters(), 3.0)#设置梯度截断阈值
        loss.backward()## 计算梯度
        optimizer.step()## 根据计算的梯度更新网络参数
        if i%100 ==0:
            print("Setp-{} Loss:{:.3f}".format(i,loss.item()))

def Eval(model, iterator):
    model.eval()
    prediction_all, crf_loss_all, y_2d_all = [],[],[]
    for i, batch in enumerate(iterator):
        _, tokens_id_l, answer_offset_l, answer_seq_label_l = batch
        prediction, crf_loss, y_2d  = model.module.forward(tokens_id_l, answer_offset_l, answer_seq_label_l)
        prediction_all.append(prediction)
        y_2d_all.append(y_2d)
        crf_loss_all.append(crf_loss.to("cpu").item())

    acc = result_metric(prediction_all, y_2d_all)
    print("Eval-Loss: {:.3f}".format(np.mean(crf_loss_all)))
    print("Eval-Result: {:.3f}".format(acc))
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--trainset", type=str, default="data/me_train.json")
    parser.add_argument("--devset", type=str, default="data/me_validation.ann.json")
    parser.add_argument("--testset", type=str, default="data/me_test.ann.json")
    parser.add_argument("--LOSS_alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default='cuda')
    if os.name == "nt":
        parser.add_argument("--model_path", type=str, default="D:\创新院\智能问答\BERT for WebQA\save_model\latest_model.pt")
        parser.add_argument("--batch_size", type=int, default=4)
    else:
        parser.add_argument("--model_path", type=str, default="BERT for WebQA/save_model/latest_model.pt")
        parser.add_argument("--batch_size", type=int, default=16)

    hp = parser.parse_args()
    print("="*20+" 超参 "+"="*20)
    for para in hp.__dict__:
        print(" " * (20 - len(para)), para, "=", hp.__dict__[para])
    train_dataset = WebQADataset(hp.trainset)
    dev_dataset = WebQADataset(hp.devset)
    test_dataset = WebQADataset(hp.testset)

    # samples_weight = train_dataset.get_samples_weight()
    # sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                #  sampler=sampler,
                                 num_workers=4,
                                 collate_fn=pad
                                 )
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad
                               )
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad
                                )


    PreModel = BertModel.from_pretrained('bert-base-chinese')

    if os.path.exists(hp.model_path):
        print('=======载入模型=======')
        model = torch.load(hp.model_path)
    else:
        print("=======初始化模型======")
        model = Net(PreModel= PreModel)
        if hp.device == 'cuda':
            model = model.cuda()
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.l2)

    if not os.path.exists(os.path.split(hp.model_path)[0]):
        os.makedirs(os.path.split(hp.model_path)[0])

    test_acc = Eval(model, test_iter)

    best_acc = max(0,test_acc )
    no_gain_rc = 0#效果不增加代数

    for epoch in range(1, hp.n_epochs + 1):
        TrainOneEpoch(model, train_iter, optimizer, hp)

        print(f"=========eval dev at epoch={epoch}=========")
        dev_acc = eval(model, dev_iter)

        print(f"=========eval test at epoch={epoch}=========")
        test_acc= eval(model, test_iter)

        if test_acc >best_acc:
            print("精度值由 {:.3f} 更新至 {:.3f} ".format(best_acc, test_acc))
            best_acc = test_acc
            print("=======保存模型=======")
            torch.save(model, hp.model_path)
            no_gain_rc = 0
        else:
            no_gain_rc = no_gain_rc+1

        # 提前终止
        if no_gain_rc > hp.early_stop:
            print("连续{}个epoch没有提升，在epoch={}提前终止".format(no_gain_rc,epoch))
            break
