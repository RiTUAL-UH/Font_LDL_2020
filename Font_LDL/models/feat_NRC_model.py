import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import config
import torch


class feat_NRC_model(nn.Module):
    def __init__(self, class_num, emotion_size, VAD_size, intensity_size):
        super(feat_NRC_model, self).__init__()
        self.class_num = class_num

        self.if_useDeepMojiFeat  = config.use_useDeepMojiFeat

        # emotion layer:
        self.emotion_size = 0
        if config.use_NRCemotion:
            self.emotion_size = emotion_size
            self.emotion_layer = nn.Sequential(
                nn.Linear(self.emotion_size, self.emotion_size)
            )

        # VAD layer:
        self.VAD_size = 0
        if config.use_NRCvad:
            self.VAD_size = VAD_size
            self.VAD_layer = nn.Sequential(
                nn.Linear(self.VAD_size, self.VAD_size)
            )

        # intensity layer:
        self.intensity_size = 0
        if config.use_NRCintensity:
            self.intensity_size = intensity_size
            self.intensity_layer = nn.Sequential(
                nn.Linear(self.intensity_size, self.intensity_size)
            )

        self.encoder_input = self.emotion_size + self.VAD_size + self.intensity_size
        print("###encoder_input: ", self.encoder_input)

        self.featureEncoder = FeatureEncoder(input_dim=self.encoder_input,
                                             hidden_dim=80)

        self.classificationHead = classificationHead(input_dim=80, inner_dim=80,
                                                                 num_classes=class_num,
                                                                 pooler_dropout=0.3,
                                                                 pooling="first")


        if torch.cuda.is_available():
            if config.use_NRCemotion:
                self.emotion_layer = self.emotion_layer.cuda()
            if config.use_NRCvad:
                self.VAD_layer = self.VAD_layer.cuda()
            if config.use_NRCintensity:
                self.intensity_layer = self.intensity_layer.cuda()

            self.featureEncoder = self.featureEncoder.cuda()
            self.classificationHead = self.classificationHead.cuda()

    def forward(self, words, mask, _, emo_tensor, intense_tensor, vad_tensor):

        if config.use_NRCemotion:
            emotion_sequence = self.emotion_layer(emo_tensor.float())
            emb_sequence = emotion_sequence

        if config.use_NRCvad:
            vad_sequence = self.VAD_layer(vad_tensor.float())
            emb_sequence = torch.cat([emb_sequence, vad_sequence], 2)

        if config.use_NRCintensity:
            intensity_sequence = self.intensity_layer(intense_tensor.float())
            emb_sequence = torch.cat([emb_sequence, intensity_sequence], 2)


        encoder_features = self.featureEncoder(emb_sequence, mask)

        logits = self.classificationHead(encoder_features, mask)
        return logits


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, self.hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
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
        lengths = torch.sum(mask,1)  # list of len of all sequences # sum up all 1 values which is equal to the lenghts of sequences
        lengths, order = lengths.sort(0, descending=True)
        recover = order.sort(0, descending=False)[1]

        sequences = sequences[order]
        packed_words = pack_padded_sequence(sequences, lengths.cpu().numpy(), batch_first=True)
        lstm_out, hidden = self.lstm(packed_words, None)

        feats, _ = pad_packed_sequence(lstm_out)
        feats = feats.permute(1, 0, 2)
        feats = feats[recover]  # feat shape: [batch_size, seq_len, hidden_dim]
        return feats

class classificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout,
                 pooling='first'):
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
            lstm_out = lstm_out.view(batch_size, max_len, 2, hidden_len // 2)
            fw_last_hn = lstm_out[range(batch_size), lengths - 1, 0]
            bw_last_hn = lstm_out[range(batch_size), 0, 1]
            last_hn = torch.cat([fw_last_hn, bw_last_hn], dim=1)
            x = last_hn
        elif self.pooling == "mean":
            x = torch.mean(lstm_out, 1)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
