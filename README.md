# Bert-for-WebQA
使用 torch 和 transformers 包从零开始搭建了一个中文阅读问答训练测试

框架 ，数据选择百度的 WebQA 问答数据集， 类似于斯坦福智能问答数据集，

使用 Bert-base-chinese 和 CRF 模型 做基础，模型可以根据需要持续更新。

输入：[‘CLS’]+Question+[‘SEP’]+Evidence 字符串

输出：0-1 序列， Evidence 中出现答案的位置为 1 ，其余为 0

数据集来自：https://pan.baidu.com/s/1QUsKcFWZ7Tg1dk_AbldZ1A 提取码：2dva

BaseLine论文：https://arxiv.org/abs/1607.06275

==================== 超参 ====================

           early_stop = 5
                   lr = 1e-05
                   l2 = 1e-05
             n_epochs = 50
               logdir = logdir
             trainset = data/me_train.json
               devset = data/me_validation.ann.json
              testset = data/me_test.ann.json
               device = cuda
                 mode = eval
           model_path = save_model/latest_model备份.pt
           model_back = save_model/back_model.pt
           batch_size = 8
           

Eval On TestData   Eval-Loss: 15.383  Eval-Result: acc = 0.796

Eval On DevData    Eval-Loss: 13.986  Eval-Result: acc = 0.795

说明：上面效果只训练了半个epoch 因为疫情在家没有服务器，用谷歌云训练的

聊天效果如下：

![image text](https://raw.githubusercontent.com/Hanlard/Bert-for-WebQA/master/问答截屏/lt.jpg)
