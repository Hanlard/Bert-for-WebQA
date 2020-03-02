# Bert-for-WebQA
用BERT在百度WebQA中文问答数据集上做阅读问答，会陆续更新算法模块。

模型用transformers和pytorch包实现，直接运行TrainAndEval.py进行训练和测试。

输入：[‘CLS’]+Question+[‘SEP’]+Evidence 字符串

输出：0 1 序列， Evidence 中出现答案的位置为 1 ，其余为 0

数据集来自：https://pan.baidu.com/s/1QUsKcFWZ7Tg1dk_AbldZ1A 提取码：2dva
