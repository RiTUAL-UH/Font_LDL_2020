import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from .crf import CRF
import numpy as np
import config

class model_glove(nn.Module):
    def __init__(self, embeddings, class_num=10):
        super(model_glove, self).__init__()

        self.use_encoder = True
        self.use_att = False
        self.if_useFeat = config.use_useDeepMojiFeat

        # word embedding:
        self.wordEmbedding = EmbeddingLayer(embeddings)

        # Encoder:
        self.encoder_input_dim = embeddings.shape[1]
        if self.use_encoder:
            self.featureEncoder = FeatureEncoder(input_dim=self.encoder_input_dim, hidden_dim=self.encoder_input_dim)


        self.classificationHead = ClassificationHead(input_dim=self.encoder_input_dim, inner_dim=200, num_classes=class_num,
                                                    pooler_dropout=0.3, pooling="first")

        if torch.cuda.is_available():
            self.wordEmbedding = self.wordEmbedding.cuda()
            self.classificationHead = self.classificationHead.cuda()
            if self.use_encoder:
                self.featureEncoder = self.featureEncoder.cuda()


    def forward(self, w_tensor, mask, _, __, ___, ____):
        emb_sequence = self.wordEmbedding(w_tensor)

        if self.use_encoder:
            encoder_features = self.featureEncoder(emb_sequence, mask)

        scores = self.classificationHead(encoder_features, mask)
        return scores # score shape: [batch_size, max_seq_len, num_labels]



class EmbeddingLayer(nn.Module):
    def __init__(self, embeddings):
        super(EmbeddingLayer, self).__init__()

        self.word_encoder = nn.Sequential(
            nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False),
            nn.Dropout(0.4)
        )

        if torch.cuda.is_available():
            self.word_encoder = self.word_encoder.cuda()

    def forward(self, w_tensor):
        return self.word_encoder(w_tensor)

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureEncoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, self.hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)

        if torch.cuda.is_available():
            self.lstm = self.lstm.cuda()
            self.dropout = self.dropout.cuda()


    def forward(self, sequences, mask):
        """
       :param sequences: sequence shape: [batch_size, seq_len, emb_dim]
       :param mask:
       :return:
        """
        lengths = torch.sum(mask, 1) # list of len of all sequences # sum up all 1 values which is equal to the lenghts of sequences
        lengths, order = lengths.sort(0, descending=True)
        recover = order.sort(0, descending=False)[1]

        sequences = sequences[order]
        packed_words = pack_padded_sequence(sequences, lengths.cpu().numpy(), batch_first=True)
        lstm_out, hidden = self.lstm(packed_words, None)

        feats, _ = pad_packed_sequence(lstm_out)
        feats = feats.permute(1, 0, 2)
        feats = feats[recover] # feat shape: [batch_size, seq_len, hidden_dim]
        return feats

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, input_dim, inner_dim, num_classes,  pooler_dropout,  pooling='first'):
        super().__init__()
        self.pooling = pooling
        self.dense1 = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, lstm_out, mask):
        batch_size = lstm_out.size(0)
        max_len = lstm_out.size(1)
        hidden_len = lstm_out.size(2)
        if self.pooling == "first":
            lengths = mask.sum(-1)
            # separate the direction of bilstm:
            lstm_out = lstm_out.view(batch_size, max_len, 2, hidden_len//2)
            fw_last_hn = lstm_out[range(batch_size), lengths-1, 0]
            bw_last_hn = lstm_out[range(batch_size), 0, 1]
            last_hn = torch.cat([fw_last_hn, bw_last_hn], dim=1)
            x = last_hn
        elif self.pooling == "mean":
            x = torch.mean(lstm_out, 1)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


