import torch
import torch.nn as nn
import torch.optim as optim
import config
from config import *
from data import Corpus, Encoder
from models.emoji_model import emoji_model
from models.feat_NRC_model import feat_NRC_model
from models.model_glove import model_glove
from models.model_bert_seq_classification import model_bert_seq_classification
from train import Trainer
import numpy as np
import os

gpu_number = config.gpu_number


if __name__ == '__main__':
    torch.manual_seed(config.torch_random_seed)
    np.random.seed(config.np_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        print("Running on GPU {}".format(gpu_number))
        torch.cuda.set_device(gpu_number)
    else:
        print("Running on CPU")

    if not os.path.exists(config.output_dir_path):
        os.makedirs(config.output_dir_path)
    if not os.path.exists(config.score_output):
        os.makedirs(config.score_output)



    print("[LOG] running . . .", version)
    batch_size = config.batch_size
    epoch =config.epoch
    corpus = Corpus.get_corpus(corpus_dir, corpus_pkl)
    encoder = Encoder.get_encoder(corpus, emb_path, encoder_pkl_path=encoder_pkl)
    encoder.encode_words(corpus)
    encoder.encode_word2_NRC_EWE(corpus)



    if base_model == "glove":
        model = model_glove(embeddings=encoder.word_emb, class_num=10)
    elif base_model == "bert_seq_classification":
        model = model_bert_seq_classification(class_num=10)
    elif base_model== "emoji":
        model = emoji_model(class_num=10)
    elif base_model == "NRCfeat":
        model = feat_NRC_model(class_num=10, emotion_size=10, VAD_size=3, intensity_size=4)

    optimizer = optim.Adam(lr=config.learning_rate, params=model.parameters())
    trainer = Trainer(epoch , batch_size, corpus)

    theLoss = nn.KLDivLoss(reduction='batchmean')
    if config.train:
        trainer.train(model, optimizer, theLoss)
    if config.test:
        trainer.predict(model, theLoss)