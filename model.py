import torch
import torch.nn as nn
from migration_model.models.CRF import CRF
import numpy as np

class Net(nn.Module):
    def __init__(self, PreModel):
        super().__init__()
        self.PreModel = PreModel
        self.hidden_size = self.PreModel.config.hidden_size
        self.labelnum = 2# 0,1
        self.CRF_fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.labelnum + 2, bias=True),
        )
        self.device = torch.device("cuda")

        kwargs = dict({'target_size': self.labelnum, 'device': self.device})
        self.CRF = CRF(**kwargs)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.fc2 = nn.Linear(self.hidden_size, 2, bias=True)

    def forward(self,tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l):

        ## 字符ID [batch_size, seq_length]
        tokens_x_2d = torch.LongTensor(tokens_id_l).to(self.device)
        token_type_ids_2d = torch.LongTensor(token_type_ids_l).to(self.device)

        # 计算sql_len 不包含[CLS]
        batch_size, seq_length = tokens_x_2d[:,1:].size()

        ## CRF答案ID [batch_size, seq_length]
        y_2d = torch.LongTensor(answer_seq_label_l).to(self.device)[:,1:]
        ## (batch_size,)
        y_IsQA_2d = torch.LongTensor(IsQA_l).to(self.device)

        if self.training: # self.training基层的外部类
            self.PreModel.train()
            emb, _ = self.PreModel(input_ids=tokens_x_2d, token_type_ids=token_type_ids_2d) #[batch_size, seq_len, hidden_size]
        else:
            self.PreModel.eval()
            with torch.no_grad():
                emb, _ = self.PreModel(input_ids=tokens_x_2d, token_type_ids=token_type_ids_2d)

        ## [CLS] for IsQA  [batch_size, hidden_size]
        cls_emb = emb[:,0,:]
        ## [batch_size, 2]
        IsQA_logits = self.fc2(cls_emb)
        IsQA_loss = self.CrossEntropyLoss.forward(IsQA_logits,y_IsQA_2d)

        ## [batch_size, 1]
        IsQA_prediction = IsQA_logits.argmax(dim=-1).unsqueeze(dim=-1)

        # CRF mask
        mask = np.ones(shape=[batch_size, seq_length], dtype=np.uint8)
        mask = torch.ByteTensor(mask).to(self.device)
        # [batch_size, seq_len, 4]

        # No [CLS]
        crf_logits = self.CRF_fc1(emb[:,1:,:])
        crf_loss = self.CRF.neg_log_likelihood_loss(feats=crf_logits, mask=mask, tags=y_2d )
        _, CRFprediction = self.CRF.forward(feats=crf_logits, mask=mask)

        return IsQA_prediction, CRFprediction, IsQA_loss, crf_loss, y_2d, y_IsQA_2d.unsqueeze(dim=-1)# (batch_size,) -> (batch_size, 1)

    def predict(self,tokens_id_l, token_type_ids_l):
        tokens_x_2d = torch.LongTensor(tokens_id_l).to(self.device)
        token_type_ids_2d = torch.LongTensor(token_type_ids_l).to(self.device)

        batch_size, seq_length = tokens_x_2d[:,1:].size()
        self.PreModel.eval()
        with torch.no_grad():
            emb, _ = self.PreModel(tokens_x_2d,token_type_ids=token_type_ids_2d)

        ## [CLS] for IsQA  [batch_size, hidden_size]
        cls_emb = emb[:,0,:]
        ## [batch_size, 2]
        IsQA_logits = self.fc2(cls_emb)
        ## [batch_size, 1]
        IsQA_prediction = IsQA_logits.argmax(dim=-1)

        # CRF mask
        mask = np.ones(shape=[batch_size, seq_length], dtype=np.uint8)
        mask = torch.ByteTensor(mask).to(self.device)
        # [batch_size, seq_len, 4]
        crf_logits = self.CRF_fc1(emb[:,1:,:])
        _, CRFprediction = self.CRF.forward(feats=crf_logits, mask=mask)

        return IsQA_prediction.to("cpu"), CRFprediction.to("cpu")