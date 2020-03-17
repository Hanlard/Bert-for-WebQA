
import json
from torch.utils import data
from transformers import BertTokenizer
import numpy as np
import jieba
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# 字符ID化
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
import jieba.posseg as pseg
class Knowledge(data.Dataset):
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


data_path = 'D:\创新院\智能问答\BERT for WebQA\data\me_validation.ann.json'
dataset = Knowledge(data_path)

Stop_path = 'D:\创新院\智能问答\BERT for WebQA\data\stop_words.txt'
with open(Stop_path, "r", encoding="gbk") as f:
    Stopword  = set(f.read().splitlines())

def del_stopword(line,Stopword,ngram=False):
    line=list(jieba.cut(line))
    new = [word  for word in line if word not in Stopword]
    if ngram :#返回2元语法
        N = len(line)
        for i, word_i in enumerate(line):
            for j in range(min(i+1,N-1),N):
                word_j = line[j]
                if word_i not in Stopword and word_j not in Stopword:
                    new.append(word_i+' '+word_j)
    return new # [w1,w2,...]


## 计算并保存语料的TF-IDF矩阵

documents = dataset.evidences[:]
corpus = [" ".join(del_stopword(e,Stopword)) for e in documents]
vectorizer = CountVectorizer()#ngram_range=(1,2)
count = vectorizer.fit_transform(corpus)
TFIDF= TfidfTransformer()
tfidf_matrix = TFIDF.fit_transform(count)
d_matrix = np.array(tfidf_matrix.toarray())
vocabulary_  = vectorizer.vocabulary_

# print("保存{}的tf-idf矩阵".format(data_path.split('\\')[-1]))
# np.save("data/me_validation_ann.npy",d_matrix)
# np.save("data/me_validation_ann_vocab.npy",dict(vectorizer.vocabulary_))
# #
#
# ## 问答
# vocabulary_ = np.load("D:\创新院\智能问答\BERT for WebQA\data\me_validation_ann_vocab.npy",allow_pickle=True).item()
# d_matrix = np.load("D:\创新院\智能问答\BERT for WebQA\data\me_validation_ann.npy")

## 测试准确度: D:\创新院\智能问答\BERT for WebQA\data\me_validation.ann.json
## 仅保留最大值 一元语法  100：0.82 1000：0.661 3000：0.5357，；二元语法 100:0.82 1000:0.682 3000：0.5579
## 保留前5个  一元语法 100:0.93 1000：0.868 3000：0.7786；二元语法 100:0.94 1000:0.873 3000：0.7876
## 加权向量
# 仅保留最大值 一元语法 100:0.86 1000：0.676
## 保留前15个 3000: 87.38%
## 保留前20个 3000：89.23%
correct = 0
test = list(range(d_matrix.shape[0]))
for  i  in test:
    q_vector = np.zeros([1, d_matrix.shape[1]])
    q1 = dataset.questions[i]
    q_list = del_stopword(q1,Stopword,ngram=False)
    for k,word in enumerate(q_list):
        if word in vocabulary_:
            q_vector[0,vocabulary_[word]] = 1 + 1/(1+k)


    dot = (np.mat(d_matrix))*(np.mat(q_vector.T))

    xu=dot.argsort(0)[-20:].tolist()# [[12], [37], [10]] 最大的五个索引

    if [i] in xu:#返回的列表中包含正确答案
        correct = correct+1
print("检索文档精度：{:.2f}%".format(100*correct/len(test)))
# while True:
#     print("输入：")
#     i= int(input())
#     q_vector = np.zeros([1, d_matrix.shape[1]])
#     q1 = dataset.questions[i]
#     q_list = del_stopword(q1,Stopword,ngram=False)
#     for k,word in enumerate(q_list):
#         if word in vocabulary_:
#             q_vector[0,vocabulary_[word]] = 1.+(1./(k+1))
#     dot = (np.mat(d_matrix))*(np.mat(q_vector.T))
#
#     xu=dot.argsort(0)[-5:].tolist()
#     xu.reverse()# [[12], [37], [10]] 最大的五个索引
#     print(xu)

# n=-1
# n=n+1
# print(dataset.questions[n])
# print(dataset.evidences[n])
# print(dataset.answer[n])