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

    def forward(self,tokens_id_l, answer_offset_l, answer_seq_label_l):

        ## 字符ID [batch_size, seq_length]
        tokens_x_2d = torch.LongTensor(tokens_id_l).to(self.device)

        batch_size, seq_length = tokens_x_2d.size()

        ## 答案ID [batch_size, seq_length]
        y_2d = torch.LongTensor(answer_seq_label_l).to(self.device)

        if self.training: # self.training基层的外部类
            self.PreModel.train()
            emb, _ = self.PreModel(tokens_x_2d) #[batch_size, seq_len, hidden_size]
        else:
            self.PreModel.eval()
            with torch.no_grad():
                emb, _ = self.PreModel(tokens_x_2d)

        # CRF mask
        mask = np.ones(shape=[batch_size, seq_length], dtype=np.uint8)
        mask = torch.ByteTensor(mask).to(self.device)
        # [batch_size, seq_len, 4]
        crf_logits = self.CRF_fc1(emb)
        crf_loss = self.CRF.neg_log_likelihood_loss(feats=crf_logits, mask=mask, tags=y_2d )
        _, prediction = self.CRF.forward(feats=crf_logits, mask=mask)
        return prediction, crf_loss, y_2d