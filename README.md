# Bert-for-WebQA
使用 torch 和 transformers 包从零开始搭建了一个中文阅读问答训练测试
框架 ，数据选择百度的 WebQA 问答数据集， 类似于斯坦福智能问答数据集，
使用 Bert-base-chinese 和 CRF 模型 做基础，模型可以根据需要持续更新。

#### 输入：[‘CLS’]+Question+[‘SEP’]+Evidence 字符串

#### 模型框架：采用多任务联合训练的方式，共两个任务：

           任务1. 使用"[CLS]"来判断两个句子是否是Quesntion-Evidence的关系；

           任务2. 使用Question+[‘SEP’]+Evidence的BERT表达 + CRF模型 进行序列标注，找出Evidence中的答案。

#### 输出：

           任务1. [batch_size,1] 的0-1 序列，1表示对应的文章中含有问题答案，0表示没有；
           
           任务2. [batch_size, seq_len] 的0-1 序列, Evidence 中出现答案的位置为 1 ，其余为 0。

#### 备注： 选择使用"[CLS]"做Quesntion-Evidence关系判断的原因是，做大规模文档检索时，通常回返回一些带有迷惑性的负样本，用"[CLS]"可以进行二次过滤。

数据集来自：https://pan.baidu.com/s/1QUsKcFWZ7Tg1dk_AbldZ1A 提取码：2dva

BaseLine论文：https://arxiv.org/abs/1607.06275

模型的谷歌云共享连接(训练好的模型)：https://drive.google.com/open?id=1KHlCnT6VEpDCvtJp8FfwMtU5_ABrYzH9

==================== 超参 ====================

           early_stop = 1
                   lr = 1e-05
                   l2 = 1e-05
             n_epochs = 5
            Negweight = 0.01
             trainset = data/me_train.json
               devset = data/me_validation.ann.json
              testset = data/me_test.ann.json
       knowledge_path = data/me_test.ann.json
        Stopword_path = data/stop_words.txt
               device = cuda
                 mode = train
           model_path = save_model/latest_model.pt
           model_back = save_model/back_model.pt
           batch_size = 16
           

Eval On TestData   Eval-Loss: 15.383  Eval-Result: acc = 0.796

Eval On DevData    Eval-Loss: 13.986  Eval-Result: acc = 0.795

说明：上面效果只训练了半个epoch 因为疫情在家没有服务器，用谷歌云训练的，设备是tesla-P100，回答一个问题平均耗时40ms。

聊天效果如下：

![image text](https://raw.githubusercontent.com/Hanlard/Bert-for-WebQA/master/问答截屏/截图.jpg)

#### 运行

训练 %run TrainAndEval.py --batch_size=8 --mode="train" --model_path='save_model/latest_model.pt'

评估 %run TrainAndEval.py --mode="eval" --model_path='save_model/latest_model.pt'

对话 %run TrainAndEval.py  --mode="demo" --model_path='save_model/latest_model.pt'

