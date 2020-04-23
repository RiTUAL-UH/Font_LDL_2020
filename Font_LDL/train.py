import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from helper import Helper
import config
from logger import Logger
helper = Helper()

logger = Logger(config.output_dir_path + 'logs')

def tensor_logging(model, info, epoch):
    for tag, value in info.items():
        logger.log_scalar(tag, value, epoch + 1)
    # Log values and gradients of the model parameters
    for tag, value in model.named_parameters():
        if value.grad is not None:
            tag = tag.replace('.', '/')
            if torch.cuda.is_available():
                logger.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

def average(lst):
    return sum(lst) / len(lst)

def to_tensor(encodings, pad_value=0, return_mask=False,type= torch.long):
    maxlen = max(map(len, encodings))
    tensor = torch.zeros(len(encodings), maxlen).long() + pad_value
    mask = torch.zeros(len(encodings), maxlen).long()
    for i, sample in enumerate(encodings):
        tensor[i, :len(sample)] = torch.tensor(sample, dtype=type)
        mask[i, :len(sample)] = 1
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        mask = mask.cuda()
    return (tensor, mask) if return_mask else tensor


def to_tensor_features(encodings, size, type):
    maxlen = max(map(len, encodings))
    tensor = []

    for i, sample in enumerate(encodings):
        seq_len = len(sample)
        padding_len = abs(seq_len - maxlen)

        pad = [[0.0]*size] * padding_len
        sample.extend(pad)
        tensor.append(sample)
    if type == "long":
        tensor_tens = torch.LongTensor(tensor)
    else:
        if type =="float":
            tensor_tens = torch.FloatTensor(tensor)
    if torch.cuda.is_available():
        tensor_tens = tensor_tens.cuda()
    return tensor_tens


class Trainer(object):
    def __init__(self,  epochs, batch_size, corpus):
        self.epochs = epochs
        self.corpus = corpus
        self.batch_size_org = batch_size
        self.batch_size = batch_size

    def batchify(self, batch_i, corpus, model):

        batch_start = batch_i * self.batch_size_org
        batch_end = batch_start + self.batch_size
        sentences = corpus.X[batch_start:batch_end]
        sentences_encoded = corpus.words_encoded[batch_start:batch_end]
        target = corpus.y[batch_start:batch_end]
        emojifeats = corpus.emoji[batch_start:batch_end]

        w_tensor, mask = to_tensor(sentences_encoded, return_mask=True)

        emo_tensor = None
        if config.use_NRCemotion:
            emo = corpus.emotion[batch_start: batch_end]
            emo_tensor = to_tensor_features(emo, 10, type= "long")


        intense_tensor = None
        if config.use_NRCintensity:
            intense = corpus.intensity[batch_start: batch_end]
            intense_tensor = to_tensor_features(intense, 4, type= "float")


        vad_tensor = None
        if config.use_NRCvad:
            vad = corpus.VAD[batch_start: batch_end]
            vad_tensor = to_tensor_features(vad, 3, type = "float")



        target = torch.tensor(target)
        emojifeats = torch.tensor(emojifeats)

        if torch.cuda.is_available():
            target = target.cuda()
            emojifeats = emojifeats.cuda()

        if config.base_model == "glove" or config.base_model == "NRCfeat" or config.base_model =="emoji":
            scores = model(w_tensor, mask, emojifeats, emo_tensor, intense_tensor, vad_tensor)
        else:
            scores = model(sentences)
        scores_flat_log_softmaxed = F.log_softmax(scores, dim=1)

        return  sentences, scores, scores_flat_log_softmaxed,  target


    def train(self, model, optimizer, theLoss):
        print("[LOG] training . . .")

        print(model)
        total_batch_train = len(self.corpus.train.y) // self.batch_size
        total_batch_dev = len(self.corpus.dev.y) // self.batch_size

        if (len(self.corpus.train.y)) % self.batch_size > 0:
            total_batch_train += 1

        if len(self.corpus.dev.y) % self.batch_size > 0:
            total_batch_dev += 1

        for epoch in range(self.epochs):
            model.train()
            self.batch_size = self.batch_size_org
            if config.if_shuffle:
                self.corpus.shuffle_train()
            print("[LOG] epoch: ", epoch)
            total_train_loss = 0
            train_all_target =[]
            train_all_score = []

            for batch_i in range(total_batch_train):
                if (batch_i == total_batch_train - 1) and (len(self.corpus.train.y) % self.batch_size > 0):
                    self.batch_size = len(self.corpus.train.y) % self.batch_size

                optimizer.zero_grad()
                sentences, scores, scores_flat_log_softmaxed, target = self.batchify(batch_i, self.corpus.train, model)

                train_loss = theLoss(scores_flat_log_softmaxed, F.softmax(target, dim=1))
                total_train_loss += train_loss.item() * self.batch_size

                train_all_target.extend(target.cpu().detach().numpy().tolist())
                train_all_score.extend(F.softmax(scores, dim=1).cpu().detach().numpy().tolist())

                train_loss.backward()
                optimizer.step()

            train_loss = total_train_loss / len(self.corpus.train.y)
            print("[LOG] train loss: {} **** ".format(train_loss))
            print("[LOG] ________________dev____________________")

            total_dev_loss = 0
            dev_all_target = []
            dev_all_score = []
            model.eval()
            self.batch_size = self.batch_size_org
            for batch_i in range(total_batch_dev):
                if (batch_i == total_batch_dev - 1) and (len(self.corpus.dev.y) % self.batch_size > 0):
                    self.batch_size = len(self.corpus.dev.y) % self.batch_size
                dev_sentences, dev_scores, dev_scores_flat_log_softmaxed, dev_target = self.batchify(batch_i, self.corpus.dev, model)
                dev_loss = theLoss(dev_scores_flat_log_softmaxed, F.softmax(dev_target, dim=1))
                total_dev_loss += dev_loss.item() * self.batch_size
                dev_all_target.extend(dev_target.cpu().detach().numpy().tolist())
                dev_all_score.extend(F.softmax(dev_scores, dim=1).cpu().detach().numpy().tolist())

            dev_loss = total_dev_loss / len(self.corpus.dev.y)
            is_best = helper.checkpoint_model(model, optimizer, config.output_dir_path, dev_loss, epoch + 1, 'min')

            print("dev loss: {} ****".format(dev_loss))
            print(config.version)
            #tensorBoard:
            info = {'training_loss': train_loss,
                    'validation_loss': dev_loss,
                    }
            tensor_logging(model, info, epoch)
            print("___________________________")

    def predict(self, model, theLoss):
        print("[LOG] starting to predict . . .  ")
        # Loading the best model:
        helper.load_saved_model(model, config.output_dir_path + 'best.pth')
        model.eval()
        self.batch_size = self.batch_size_org
        total_test_loss =0
        total_batch_test = len(self.corpus.test.y) // self.batch_size
        if len(self.corpus.test.y) % self.batch_size > 0:
            total_batch_test += 1
        test_all_target =[]
        test_all_score =[]
        all_sents =[]
        for batch_i in range(total_batch_test):
            if (batch_i == total_batch_test - 1) and (len(self.corpus.test.y) % self.batch_size > 0):
                self.batch_size = len(self.corpus.test.y) % self.batch_size
            sentences, scores, scores_flat_log_softmaxed, target = self.batchify(batch_i, self.corpus.test, model)
            loss = theLoss(scores_flat_log_softmaxed, F.softmax(target, 1))
            total_test_loss += loss.item() * self.batch_size
            test_all_target.extend(target.cpu().detach().numpy().tolist())
            test_all_score.extend(F.softmax(scores, 1).cpu().detach().numpy().tolist())
            all_sents.extend(sentences)

        test_loss = total_test_loss / len(self.corpus.test.y)
        print("[LOG] test loss: {} ****".format(test_loss))

        #writing test scores
        with open(os.path.join(config.score_output,"score_test.txt"), "w") as f:
            for i in range(len(test_all_score)):
                f.write(all_sents[i] + "\t\t" + str(test_all_score[i][0]) + "\t" + str(test_all_score[i][1]) + "\t" + str(test_all_score[i][2]) + "\t" + str(test_all_score[i][3])
                        + "\t" + str(test_all_score[i][4]) + "\t" + str(test_all_score[i][5]) + "\t" + str(test_all_score[i][6]) + "\t" + str(test_all_score[i][7])
                        + "\t" + str(test_all_score[i][8]) + "\t" + str(test_all_score[i][9]) + "\n")
        f.close()
        with open(os.path.join(config.score_output,"target_test.txt"), "w") as f2:
            for i in range(len(test_all_target)):
                f2.write(all_sents[i] + "\t\t" + str(test_all_target[i][0]) + "\t" + str(test_all_target[i][1]) + "\t" + str(test_all_target[i][2])
                         + "\t" + str(test_all_target[i][3]) + "\t" + str(test_all_target[i][4]) + "\t" + str(test_all_target[i][5])
                         + "\t" + str(test_all_target[i][6]) + "\t" + str(test_all_target[i][7]) + "\t" + str(test_all_target[i][8]) + "\t" + str(test_all_target[i][9]) + "\n")

        f2.close()
        print(config.version)


















