import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import config
import torch


class emoji_model(nn.Module):
    def __init__(self, class_num):
        super(emoji_model, self).__init__()
        self.class_num = class_num
        self.classificationHead = classificationHead(inner_dim=300,
                                                         num_classes=class_num, pooler_dropout=0.3,
                                                         feat_dim=2304)
        if torch.cuda.is_available():
            self.classificationHead = self.classificationHead.cuda()

    def forward(self, words, mask, emojifeats, _, __, ___):
        logits = self.classificationHead(emojifeats)
        return logits



class classificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, feat_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(feat_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim , num_classes)

    def forward(self,  extra_features):
        feat = self.dense(extra_features)
        x = self.dropout(feat)
        x = self.out_proj(x)
        return x

