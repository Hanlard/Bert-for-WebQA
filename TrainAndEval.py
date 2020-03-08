import os
import json
from torch.utils import data
from transformers import BertTokenizer , BertModel
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
from model import Net
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from DocumentRetrieval import Knowledge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import random
import jieba
# 字符ID化
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

parser = argparse.ArgumentParser()
parser.add_argument("--early_stop", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--l2", type=float, default=1e-5)
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--Negweight", type=float, default=0.01)
parser.add_argument("--trainset", type=str, default="data/me_train.json")
parser.add_argument("--devset", type=str, default="data/me_validation.ann.json")
parser.add_argument("--testset", type=str, default="data/me_test.ann.json")
parser.add_argument("--knowledge_path", type=str, default="data/me_test.ann.json")
parser.add_argument("--Stopword_path",type=str, default= 'data/stop_words.txt')
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--mode", type=str, default='train')  # eval / demo / train / QA
if os.name == "nt":
    parser.add_argument("--model_path", type=str, default="D:\创新院\智能问答\BERT for WebQA\save_model\latest_model.pt")
    parser.add_argument("--model_back", type=str, default="D:\创新院\智能问答\BERT for WebQA\save_model\\back_model.pt")
    parser.add_argument("--batch_size", type=int, default=4)
else:
    parser.add_argument("--model_path", type=str, default="save_model/latest_model.pt")
    parser.add_argument("--model_back", type=str, default="save_model/back_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)

hp = parser.parse_args()

class WebQADataset(data.Dataset):
    def __init__(self, fpath):
        self.hp = hp
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
        shuffled_l = list(zip(self.questions, self.evidences, self.answer))
        random.shuffle(shuffled_l)
        self.questions[:], self.evidences[:], self.answer[:] = zip(*shuffled_l)

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

        tokens = tokenizer.tokenize('[CLS]'+q+'[SEP]'+e)# list
        if len(tokens)>512:
            tokens=tokens[:512]
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0 if i <= tokens_id .index(102) else 1 for i in range(len(tokens_id ))]
        answer_offset = (-1, -1)
        IsQA = 0
        answer_seq_label = len(tokens_id) * [0]
        if a != 'no_answer':
            answer_tokens = tokenizer.tokenize(a)
            answer_offset = self.FindOffset(tokens, answer_tokens)#有肯能返回False
            if answer_offset:#在原文中找到答案
                answer_seq_label[answer_offset[0]:answer_offset[1]] = [1]*(len(answer_tokens))
                IsQA = 1
            else:# self.FindOffset 返回False
                answer_offset = (-1, -1)
        return tokens, tokens_id, token_type_ids, answer_offset, answer_seq_label, IsQA

    def get_samples_weight(self,Negweight):
        samples_weight = []
        for ans in self.answer:
            if ans != 'no_answer':
                samples_weight.append(1.0)
            else:
                samples_weight.append(Negweight)
        return np.array(samples_weight)

def pad(batch):
    tokens_l, tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l= list(map(list, zip(*batch)))
    maxlen = np.array([len(sen) for sen in tokens_l]).max()
    ### pad和截断
    for i in range(len(tokens_l)):
        tokens = tokens_l[i]
        tokens_id= tokens_id_l[i]
        # answer_offset = answer_offset_l[i]
        answer_seq_label = answer_seq_label_l[i]
        token_type_ids = token_type_ids_l[i]
        tokens_l[i] = tokens + (maxlen - len(tokens))*['[PAD]']
        token_type_ids_l[i] = token_type_ids + (maxlen - len(tokens))*[1]
        tokens_id_l[i] =tokens_id + (maxlen - len(tokens))*tokenizer.convert_tokens_to_ids(['[PAD]'])
        answer_seq_label_l[i] = answer_seq_label + [0]*(maxlen - len(tokens))
    return tokens_l, tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l

def result_metric(prediction_all, y_2d_all):
    total_num=0
    toral_cur=0
    for prediction, y_2d in zip(prediction_all, y_2d_all):
        batch_size,seq_len = prediction.size()
        currect = torch.sum(torch.sum(prediction == y_2d, dim=1)==seq_len).to("cpu").item()
        toral_cur = toral_cur + currect
        total_num = total_num + batch_size
    return toral_cur/total_num

def TrainOneEpoch(model, train_iter, dev_iter, test_iter, optimizer, hp):
    model.train()
    CRFprediction_all, CRFloss_all, IsQAloss_all, y_CRF_all, IsQA_prediction_all, y_IsQA_all= [],[],[],[],[],[]

    best_acc = 0
    for i, batch in enumerate(tqdm(train_iter)):
        _, tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l = batch
        optimizer.zero_grad()
        IsQA_prediction, CRFprediction, IsQA_loss, crf_loss, y_2d, y_IsQA_2d  = model.module.forward(tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l)
        ## CRF
        CRFprediction_all.append(CRFprediction)
        y_CRF_all.append(y_2d)

        ## IsQA
        IsQA_prediction_all.append(IsQA_prediction)

        y_IsQA_all.append(y_IsQA_2d)

        # loss
        CRFloss_all.append(crf_loss.to("cpu").item())
        IsQAloss_all.append(IsQA_loss.to("cpu").item())
        loss = IsQA_loss + crf_loss

        # nn.utils.clip_grad_norm_(model.parameters(), 3.0)#设置梯度截断阈值
        loss.backward()## 计算梯度
        optimizer.step()## 根据计算的梯度更新网络参数


        if i % 100 == 0 and i > 0:
            accCRF = result_metric(CRFprediction_all, y_CRF_all)
            accIsQA = result_metric(IsQA_prediction_all, y_IsQA_all)

            print("<Last 100 Steps MeanValue> Setp-{} IsQA-Loss: {:.3f} CRF-Loss: {:.3f}  "
                  "CRF-Result: accCRF = {:.3f}  IsQA-Result: accIsQA = {:.3f}"
                  .format(i,np.mean(IsQAloss_all), np.mean(CRFloss_all),accCRF,accIsQA))

        if i % 1000 == 0:
            print("Eval on Devset...")
            accIsQA, accCRF = Eval(model, dev_iter)
            if accIsQA * accCRF > best_acc:
                best_acc = accIsQA * accCRF
                if i>1000:
                    print("Devdata 精度提升 备份模型至{}".format(hp.model_back))
                    torch.save(model, hp.model_back)
            model.train()


def Eval(model, iterator):

    model.eval()
    CRFprediction_all, CRFloss_all, IsQAloss_all, y_CRF_all, IsQA_prediction_all, y_IsQA_all= [],[],[],[],[],[]
    for i, batch in enumerate(iterator):
        _, tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l = batch
        IsQA_prediction, CRFprediction, IsQA_loss, crf_loss, y_2d, y_IsQA_2d  = model.module.forward(tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l,IsQA_l)

        ## CRF
        CRFprediction_all.append(CRFprediction)
        y_CRF_all.append(y_2d)

        ## IsQA
        IsQA_prediction_all.append(IsQA_prediction)
        y_IsQA_all.append(y_IsQA_2d)


        CRFloss_all.append(crf_loss.to("cpu").item())
        IsQAloss_all.append(IsQA_loss.to("cpu").item())

    accCRF = result_metric(CRFprediction_all, y_CRF_all)
    accIsQA = result_metric(IsQA_prediction_all, y_IsQA_all)

    print("<本次评估结果> IsQA-Loss: {:.3f} CRF-Loss: {:.3f} CRF-Result: accCRF = {:.3f} "
          "IsQA-Result: accIsQA = {:.3f}".format(np.mean(IsQAloss_all), np.mean(CRFloss_all), accCRF, accIsQA))

    return accIsQA, accCRF

def Demo(model, q, e):
    tokens = tokenizer.tokenize('[CLS]' + q + '[SEP]' + e)  # list
    if len(tokens) > 512:
        tokens = tokens[:512]
    tokens_id = [tokenizer.convert_tokens_to_ids(tokens)]#[[101,...,102,...]]
    token_type_ids = [0 if i <= tokens_id[0].index(102) else 1 for i in range(len(tokens_id))]
    IsQA_prediction, CRFprediction = model.module.predict(tokens_id,token_type_ids)
    CRFprediction = CRFprediction.numpy()[0]
    IsQA_prediction = IsQA_prediction.numpy()[0]
    answer = ""
    if IsQA_prediction==1:
        for i in range(len(tokens)):
            if CRFprediction[i].item()==1:
                answer = answer + tokens[i]
    return answer

# def prepare_knowledge(knowledge_path,Stopword_path):
#     def del_stopword(line, Stopword, ngram=False):
#         line = list(jieba.cut(line))
#         new = [word for word in line if word not in Stopword]
#         if ngram:  # 返回2元语法
#             N = len(line)
#             for i, word_i in enumerate(line):
#                 for j in range(min(i + 1, N - 1), N):
#                     word_j = line[j]
#                     if word_i not in Stopword and word_j not in Stopword:
#                         new.append(word_i + ' ' + word_j)
#         return new  # [w1,w2,...]
#     print("正在准备知识库...")
#     dataset = Knowledge(knowledge_path)
#     with open(Stopword_path, "r", encoding="gbk") as f:
#         Stopword = set(f.read().splitlines())
#     documents = dataset.evidences
#     corpus = [" ".join(del_stopword(e, Stopword)) for e in documents]
#     vectorizer = CountVectorizer()  # ngram_range=(1,2)
#     count = vectorizer.fit_transform(corpus)
#     # 计算TF-IDF向量
#     TFIDF = TfidfTransformer()
#     tfidf_matrix = TFIDF.fit_transform(count)
#     d_matrix = np.array(tfidf_matrix.toarray())
#     vocabulary_ = vectorizer.vocabulary_
#     return d_matrix, vocabulary_, del_stopword, Stopword, dataset
#
# def QA(model, question, xu, knowledge, q_list):
#     xu.reverse()# 按相关度从大到小
#     result = {}
#     for index in xu:#[[1],[2],[3],...]
#         q=question
#         e=knowledge.evidences[index[0]]
#
#         answer=Demo(model,q,e)
#         print("answer:",answer)
#         print("evidence:",e)
#         if answer in result:
#             result[answer] = result[answer] + 1
#         else:
#             result[answer] = 1
#     return result



if __name__ == "__main__":
    print("="*20+" 超参 "+"="*20)
    for para in hp.__dict__:
        print(" " * (20 - len(para)), para, "=", hp.__dict__[para])

    if hp.mode == "train":
        train_dataset = WebQADataset(hp.trainset)
        dev_dataset = WebQADataset(hp.devset)
        test_dataset = WebQADataset(hp.testset)

        samples_weight = train_dataset.get_samples_weight(hp.Negweight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        train_iter = data.DataLoader(dataset=train_dataset,
                                     batch_size=hp.batch_size,
                                     shuffle=False,
                                     sampler=sampler,
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

        print("First Eval On TestData")
        accIsQA, accCRF = Eval(model, test_iter)

        best_acc = max(0, accIsQA*accCRF )
        no_gain_rc = 0#效果不增加代数

        for epoch in range(1, hp.n_epochs + 1):
            print(f"=========TRAIN and EVAL at epoch={epoch}=========")
            TrainOneEpoch(model, train_iter, dev_iter,test_iter, optimizer, hp)

            # print(f"=========eval dev at epoch={epoch}=========")
            # dev_acc = eval(model, dev_iter)

            print(f"=========eval test at epoch={epoch}=========")
            accIsQA, accCRF = Eval(model, test_iter)

            if accIsQA*accCRF >best_acc:
                print("精度值由 {:.3f} 更新至 {:.3f} ".format(best_acc, accIsQA*accCRF))
                best_acc = accIsQA*accCRF
                print("=======保存模型=======")
                torch.save(model, hp.model_path)
                no_gain_rc = 0
            else:
                no_gain_rc = no_gain_rc+1

            # 提前终止
            if no_gain_rc > hp.early_stop:
                print("连续{}个epoch没有提升，在epoch={}提前终止".format(no_gain_rc,epoch))
                break
    elif hp.mode =="eval":

        dev_dataset = WebQADataset(hp.devset)
        test_dataset = WebQADataset(hp.testset)

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

        if os.path.exists(hp.model_path):
            print('=======载入模型=======')
            model = torch.load(hp.model_path)
            print("Eval On TestData")
            _,_ = Eval(model, test_iter)
            print("Eval On DevData")
            _,_ = Eval(model, dev_iter)
        else:
            print("没有可用模型！")

    elif hp.mode =="demo":
        if os.path.exists(hp.model_path):
            print('=======载入模型=======')
            model = torch.load(hp.model_path)
            ques_num=1
            while True:
                try:
                    print("请输入问题-{}:".format(ques_num))
                    question = input()
                    if question == "OVER":
                        print("问答结束！")
                        break
                    print("请输入文章：")
                    evidence = input()
                    # print("正在解析...")
                    start = time.time()
                    answer = Demo(model,question,evidence)
                    end = time.time()
                    if answer:
                        print("问题-{}的答案是：{}".format(ques_num,answer))
                        print("耗时:{:.2f}毫秒".format((end-start)*1e3))
                    else:
                        print("文章中没有答案")
                    ques_num = ques_num + 1
                except:
                    print("问答结束！")
                    break
        else:
            print("没有可用模型！")
    # elif hp.mode =="QA":
    #     if os.path.exists(hp.model_path):
    #         print('=======载入模型=======')
    #         model = torch.load(hp.model_path)
    #         ques_num=1
    #         #准备知识库
    #         d_matrix, vocabulary_, del_stopword, Stopword, knowledge = prepare_knowledge(hp.knowledge_path,hp.Stopword_path)
    #         while True:
    #             # try:
    #             print("请输入问题-{}:".format(ques_num))
    #             question = input()
    #             if question == "OVER":
    #                 print("问答结束！")
    #                 break
    #             # 创建问句tf-idf向量
    #             q_vector = np.zeros([1, d_matrix.shape[1]])
    #             q_list = del_stopword(question,Stopword,ngram=False)
    #             for word in q_list:
    #                 if word in vocabulary_:
    #                     q_vector[0,vocabulary_[word]] = 1.
    #             dot = (np.mat(d_matrix))*(np.mat(q_vector.T))
    #             xu=dot.argsort(0)[-5:].tolist()# [[12], [37], [10]] 最大的五个索引
    #
    #             start = time.time()
    #             answer = QA(model, question, xu, knowledge, q_list)
    #             end = time.time()
    #
    #             print("问题-{}的答案是:".format(ques_num))
    #             print(answer)
    #             print("耗时:{:.2f}毫秒".format((end-start)*1e3))
    #
    #             ques_num = ques_num + 1
    #             # except:
    #             #     print("问答结束！")
    #             #     break
    #     else:
    #         print("没有可用模型！")
    else:
        print("--mode请选择：train/eval/demo, 请注意拼写")



