import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import config
import torch
from transformers import *

class model_bert_seq_classification(nn.Module):
    def __init__(self, class_num):
        super(model_bert_seq_classification, self).__init__()
        self.class_num = class_num
        tokenizer_class = BertTokenizer
        model_class = BertForSequenceClassification
        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)


        self.if_useFeat = config.use_useDeepMojiFeat
        self.pooling = "first"


        self.classification_input_size = 768
        self.model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True,  num_labels = 10)

        self.classificationHead = ClassificationHead(input_dim=self.classification_input_size, inner_dim=300,
                                                     num_classes=class_num,  pooler_dropout=0.3,
                                                     pooling=self.pooling)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.classificationHead = self.classificationHead.cuda()


    def forward(self, words_batch):

        tokens = self.tokenizer.batch_encode_plus(words_batch, pad_to_max_length=True)

        all_input_ids = torch.tensor(tokens["input_ids"])
        all_input_mask = torch.tensor(tokens["attention_mask"])
        all_segment_ids = torch.tensor(tokens["token_type_ids"])

        if torch.cuda.is_available():
            all_input_ids = all_input_ids.cuda()
            all_input_mask = all_input_mask.cuda()
            all_segment_ids = all_segment_ids.cuda()

        with torch.no_grad():
            _, hidden_states = self.model(input_ids=all_input_ids, attention_mask= all_input_mask, token_type_ids = all_segment_ids) [-2:] # Models outputs are now tuples
            hidden_state = hidden_states[-1]

        logits = self.classificationHead(hidden_state)

        return logits

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout, pooling):
        super().__init__()

        self.pooling = pooling
        self.dense1 = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self,  last_hidden_states):
        if self.pooling == "first":
            x = last_hidden_states[:, 0, :]
        elif self.pooling == "mean":
            x = torch.mean(last_hidden_states ,1)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x
