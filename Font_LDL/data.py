import os
import pandas as pd
import pickle
import numpy as np
import config
from nltk.corpus import wordnet as wn
from collections import Counter
from nltk.stem import PorterStemmer
import spacy
tokenizer = spacy.load("en_core_web_sm")
ps = PorterStemmer()
import random
random.seed(config.random_random_seed)

def read_text_embeddings(filename):
    embeddings = []
    word2index = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            word2index[line[0]] = i
            embeddings.append(list(map(float, line[1:])))
    assert len(word2index) == len(embeddings)
    return word2index, np.array(embeddings)

def flatten(elems):
    return [e for elem in elems for e in elem]

class Encoder(object):
    def __init__(self, corpus, emb_path):
        self.word2index, self.word_emb = self.get_pretrain_embeddings(emb_path, corpus.get_word_vocab())
        self.index2word = {i: w for w, i in self.word2index.items()}
        self.word2emotion, self.emotion_lst = self.get_emotion_dict(config.emotion_path, corpus.get_word_vocab())
        self.word2VAD, self.VAD_lst = self.get_VAD_dict(config.VAD_path, corpus.get_word_vocab())
        self.word2intensity, self.intensity_lst = self.get_intensity_dict(config.intensity_path, corpus.get_word_vocab())

    def encode_word2_NRC_EWE(self, corpus):
        #NRC:
        self.encode_word2emotion(corpus)
        self.encode_word2VAD(corpus)
        self.encode_word2intensity(corpus)

    def encode_words(self, corpus):
        corpus.train.words_encoded = [self.encode(self.word2index, sample) for sample in corpus.train.X_tokenized]
        corpus.dev.words_encoded = [self.encode(self.word2index, sample) for sample in corpus.dev.X_tokenized]
        corpus.test.words_encoded = [self.encode(self.word2index, sample) for sample in corpus.test.X_tokenized]


    def encode_word2emotion(self, corpus):
        corpus.train.emotion = [self.encode(self.word2emotion, sample) for sample in corpus.train.X_tokenized]
        corpus.dev.emotion = [self.encode(self.word2emotion, sample) for sample in corpus.dev.X_tokenized]
        corpus.test.emotion = [self.encode(self.word2emotion, sample) for sample in corpus.test.X_tokenized]

    def encode_word2VAD(self, corpus):
        corpus.train.VAD = [self.encode(self.word2VAD, sample) for sample in corpus.train.X_tokenized]
        corpus.dev.VAD = [self.encode(self.word2VAD, sample) for sample in corpus.dev.X_tokenized]
        corpus.test.VAD = [self.encode(self.word2VAD, sample) for sample in corpus.test.X_tokenized]

    def encode_word2intensity(self, corpus):
        corpus.train.intensity = [self.encode(self.word2intensity, sample) for sample in corpus.train.X_tokenized]
        corpus.dev.intensity = [self.encode(self.word2intensity, sample) for sample in corpus.dev.X_tokenized]
        corpus.test.intensity = [self.encode(self.word2intensity, sample) for sample in corpus.test.X_tokenized]


    def encode(self, elem2index, elems):
        return [elem2index[elem] for elem in elems]

    @staticmethod
    def get_encoder(corpus, emb_path, encoder_pkl_path):
        if os.path.exists(encoder_pkl_path):
            encoder = Encoder.load(encoder_pkl_path)
        else:
            encoder = Encoder(corpus, emb_path)
            encoder.save(encoder_pkl_path)

        Encoder.print_stats(encoder)

        return encoder

    def print_stats(self):
        print('[LOG]')
        print("[LOG] Word vocab size: {}".format(len(self.word2index)))

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def get_pretrain_embeddings(self, filename, vocab):
        # making embeddings for GloVe
        assert len(vocab) == len(set(vocab)), "The vocabulary contains repeated words"

        w2i, emb = read_text_embeddings(filename)
        word2index = {'+pad+': 0, '+unk+': 1}
        embeddings = np.zeros((len(vocab) + 2, emb.shape[1]))

        scale = np.sqrt(3.0 / emb.shape[1])
        embeddings[word2index['+unk+']] = np.random.uniform(-scale, scale, (1, emb.shape[1]))

        perfect_match = 0
        case_match = 0
        no_match = 0

        for i in range(len(vocab)):
            word = vocab[i]
            index = len(word2index)  # do not use i because word2index has predefined tokens

            word2index[word] = index
            if word in w2i:
                embeddings[index] = emb[w2i[word]]
                perfect_match += 1
            elif word.lower() in w2i:
                embeddings[index] = emb[w2i[word.lower()]]
                case_match += 1
            else:
                embeddings[index] = np.random.uniform(-scale, scale, (1, emb.shape[1]))
                no_match += 1
        print("[LOG] Word embedding stats -> Perfect match: {}; Case match: {}; No match: {}".format(perfect_match,
                                                                                                     case_match,
                                                                                                     no_match))
        return word2index, embeddings

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines

    def get_emotion_dict(self, filename, vocab):
        emotion_lst = []
        vocab_emotion_dict = {}

        emotions ={"anger":0, "anticipation":1, "disgust":2, "fear":3, "joy":4, "negative":5, "positive":6,	"sadness":7, "surprise":8, "trust":9}
        emotion_lst.extend(list(emotions.keys()))
        emotion_dict_all ={}
        emotion_dict_all_stemmed = {}
        lines = self.read_lines(filename) + ['']
        # make a dict for all words:
        for line in lines:
            if line:
                word, emotion, value = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]
                word = word.lower()
                if word in emotion_dict_all:
                    emotion_dict_all[word][emotions[emotion]] += int(value)
                    emotion_dict_all_stemmed[ps.stem(word)][emotions[emotion]] += int(value)
                else:
                    emotion_dict_all[word] = [0]*len(emotions)
                    emotion_dict_all[word][emotions[emotion]] += int(value)

                    emotion_dict_all_stemmed[ps.stem(word)] = [0] * len(emotions)
                    emotion_dict_all_stemmed[ps.stem(word)][emotions[emotion]] += int(value)

        emotion_seen_words, emotion_seen_stemed_dict, emotion_in_wnet, emotion_not_seen_words = 0, 0 ,0, 0
        for w in vocab:
            w_low = w.lower()

            if w_low in emotion_dict_all:
                vocab_emotion_dict[w] = emotion_dict_all[w_low]
                emotion_seen_words +=1


            elif w_low in emotion_dict_all_stemmed:
                vocab_emotion_dict[w] = emotion_dict_all_stemmed[w_low]
                emotion_seen_stemed_dict +=1


            elif ps.stem(w_low) in emotion_dict_all:
                vocab_emotion_dict[w] = emotion_dict_all[ps.stem(w_low)]
                emotion_seen_words += 1


            elif ps.stem(w_low) in emotion_dict_all_stemmed:
                vocab_emotion_dict[w] = emotion_dict_all_stemmed[ps.stem(w_low)]
                emotion_seen_stemed_dict +=1


            elif self.find_similar_words(w_low, 0) in emotion_dict_all:
                vocab_emotion_dict[w] = emotion_dict_all[self.find_similar_words(w_low, 0)]
                emotion_in_wnet +=1


            elif self.find_similar_words(ps.stem(w_low), 0) in emotion_dict_all:
                vocab_emotion_dict[w] = emotion_dict_all[self.find_similar_words(ps.stem(w_low), 0)]
                emotion_in_wnet +=1


            elif self.find_similar_words(w_low, 0) in emotion_dict_all_stemmed:
                vocab_emotion_dict[w] = emotion_dict_all_stemmed[self.find_similar_words(w_low, 0)]
                emotion_in_wnet +=1


            elif self.find_similar_words(ps.stem(w_low), 0) in emotion_dict_all_stemmed:
                vocab_emotion_dict[w] = emotion_dict_all_stemmed[self.find_similar_words(ps.stem(w_low), 0)]
                emotion_in_wnet +=1


            elif self.find_similar_words(w_low, 1) in emotion_dict_all_stemmed:
                vocab_emotion_dict[w] = emotion_dict_all_stemmed[self.find_similar_words(w_low, 1)]
                emotion_in_wnet +=1


            elif self.find_similar_words(w_low, 2) in emotion_dict_all_stemmed:
                vocab_emotion_dict[w] = emotion_dict_all_stemmed[self.find_similar_words(w_low, 2)]
                emotion_in_wnet +=1

            elif self.find_similar_words(w_low, 3) in emotion_dict_all_stemmed:
                vocab_emotion_dict[w] = emotion_dict_all_stemmed[self.find_similar_words(w_low, 3)]
                emotion_in_wnet +=1

            else:
                #print("NOT find this: ", w)
                vocab_emotion_dict[w] = [-1]*len(emotions)
                emotion_not_seen_words +=1

        print("[LOG] ** EMOTION seen words: {}, emotion_seen_stemed_dict: {}, emotion in Wnet:{}  emotion NOT seen words: {} ".format( emotion_seen_words, emotion_seen_stemed_dict, emotion_in_wnet, emotion_not_seen_words))

        return vocab_emotion_dict, emotion_lst

    def get_VAD_dict(self, filename, vocab):
        VAD_lst = []
        vocab_VAD_dict = {}

        VADs ={"Valence":0,	"Arousal":1, "Dominance":2}
        VAD_lst.extend(list(VADs.keys()))
        VAD_dict_all ={}
        VAD_dict_all_stemmed = {}
        lines = self.read_lines(filename) + ['']
        # make a dict for all words:
        for line in lines:
            if line:
                word, valence, arosal, dominance = line.split("\t")[0], float(line.split("\t")[1]), float(line.split("\t")[2]), float(line.split("\t")[3])
                # print("words: ", words)
                VAD_dict_all[word] = [valence, arosal, dominance]
                VAD_dict_all_stemmed[ps.stem(word)] = [valence, arosal, dominance]

        # make a dict for all the vocabs:
        VAD_seen_words, VAD_seen_stemed, VAD_in_wnet, VAD_not_seen_words = 0, 0, 0, 0
        for w in vocab:
            if w.lower() in VAD_dict_all:
                vocab_VAD_dict[w] = VAD_dict_all[w.lower()]
                VAD_seen_words += 1
            elif ps.stem(w.lower()) in VAD_dict_all:
                vocab_VAD_dict[w] = VAD_dict_all[ps.stem(w.lower())]
                VAD_seen_stemed += 1
            elif ps.stem(w.lower()) in VAD_dict_all_stemmed:
                vocab_VAD_dict[w] = VAD_dict_all_stemmed[ps.stem(w.lower())]
                VAD_seen_stemed += 1
            elif self.find_similar_words(w, 0) in VAD_dict_all:
                vocab_VAD_dict[w] = VAD_dict_all[self.find_similar_words(w, 0)]
                VAD_in_wnet += 1
            elif ps.stem(self.find_similar_words(w, 0)) in VAD_dict_all_stemmed:
                vocab_VAD_dict[w] = VAD_dict_all_stemmed[ps.stem(self.find_similar_words(w, 0))]
                VAD_in_wnet += 1
            elif self.find_similar_words(ps.stem(w), 0) in VAD_dict_all_stemmed:
                vocab_VAD_dict[w] = VAD_dict_all_stemmed[self.find_similar_words(ps.stem(w), 0)]
                VAD_in_wnet += 1
            elif ps.stem(self.find_similar_words(w, 1)).lower() in VAD_dict_all:
                vocab_VAD_dict[w] = VAD_dict_all[ps.stem(self.find_similar_words(w, 1)).lower()]
                VAD_in_wnet += 1
            elif self.find_similar_words(w, 2) in VAD_dict_all:
                vocab_VAD_dict[w] = VAD_dict_all[self.find_similar_words(w, 2)]
                VAD_in_wnet += 1
            elif self.find_similar_words(w, 3) in VAD_dict_all:
                vocab_VAD_dict[w] = VAD_dict_all[self.find_similar_words(w, 3)]
                VAD_in_wnet += 1
            else:
                vocab_VAD_dict[w] = [0]*len(VADs)
                VAD_not_seen_words +=1
        print("[LOG] ***VAD seen words: {}, VAD seen stemed: {}, VAD in Wnet:{}  VAD NOT seen words: {} ".format(
            VAD_seen_words, VAD_seen_stemed, VAD_in_wnet, VAD_not_seen_words))

        return vocab_VAD_dict, VAD_lst

    def get_intensity_dict(self, filename, vocab):
        intensity_dict = {}
        intensity_dict_stemmed = {}
        vocab_intensity_dict ={}
        intensities = {"anger": 0, "fear": 1, "sadness": 2, "joy":3}
        intensity_lst =list(intensities.keys())
        lines = self.read_lines(filename) + ['']
        for line in lines:
            if line:
                word, score, intensity = line.split("\t")[0], float(line.split("\t")[1]), line.split("\t")[2]
                if word not in intensity_dict:
                    intensity_dict[word] =[0.0, 0.0, 0.0, 0.0]
                    intensity_dict_stemmed[word] = [0.0, 0.0, 0.0, 0.0]
                if intensity =="anger":
                    intensity_dict[word][0] = score
                    intensity_dict_stemmed[word][0] = score
                elif intensity =="fear":
                    intensity_dict[word][1] = score
                    intensity_dict_stemmed[word][1] = score
                elif intensity =="sadness":
                    intensity_dict[word][2] = score
                    intensity_dict_stemmed[word][2] = score
                elif intensity =="joy":
                    intensity_dict[word][3] = score
                    intensity_dict_stemmed[word][3] = score
                else:
                    print("****************ERROR!")
                    raise NameError("name error!")


        # make a dict for all the vocabs:
        int_seen_words, int_seen_stemed, int_in_wnet, int_not_seen_words = 0, 0, 0, 0
        for w in vocab:
            if w.lower() in intensity_dict:
                vocab_intensity_dict[w] = intensity_dict[w.lower()]
                int_seen_words += 1
                #print("SEEN Intensity: ", w)

            elif ps.stem(w.lower()) in intensity_dict:
                vocab_intensity_dict[w] = intensity_dict[ps.stem(w.lower())]
                int_seen_stemed += 1
            elif ps.stem(w.lower()) in intensity_dict_stemmed:
                vocab_intensity_dict[w] = intensity_dict_stemmed[ps.stem(w.lower())]
                int_seen_stemed += 1

            elif self.find_similar_words(w, 0) in intensity_dict:
                vocab_intensity_dict[w] = intensity_dict[self.find_similar_words(w, 0)]
                int_in_wnet += 1
            elif ps.stem(self.find_similar_words(w, 0)) in intensity_dict:
                vocab_intensity_dict[w] = intensity_dict[ps.stem(self.find_similar_words(w, 0))]
                int_in_wnet += 1
            elif ps.stem(self.find_similar_words(w, 0)) in intensity_dict_stemmed:
                vocab_intensity_dict[w] = intensity_dict_stemmed[ps.stem(self.find_similar_words(w, 0))]
                int_in_wnet += 1
            elif self.find_similar_words(w, 1) in intensity_dict:
                vocab_intensity_dict[w] = intensity_dict[self.find_similar_words(w, 1)]
                int_in_wnet += 1
            elif ps.stem(self.find_similar_words(w, 1)).lower() in intensity_dict_stemmed:
                vocab_intensity_dict[w] = intensity_dict_stemmed[ps.stem(self.find_similar_words(w, 1)).lower()]
                int_in_wnet += 1
            elif self.find_similar_words(w, 2) in intensity_dict:
                vocab_intensity_dict[w] = intensity_dict[self.find_similar_words(w, 2)]
                int_in_wnet += 1
            elif self.find_similar_words(w, 3) in intensity_dict:
                vocab_intensity_dict[w] = intensity_dict[self.find_similar_words(w, 3)]
                int_in_wnet += 1
            else:
                vocab_intensity_dict[w] = [0] * len(intensities)
                int_not_seen_words+=1
                #print("Not SEEN Intensity: ", w)
        print("[LOG] ***INTENSITY seen words: {}, intensity seen stemed: {}, intensity in Wnet:{}  intensity NOT seen words: {} ".format(
            int_seen_words, int_seen_stemed, int_in_wnet, int_not_seen_words))
        return vocab_intensity_dict , intensity_lst



    def find_similar_words(self, word, i):
        if wn.synsets(word):
            syn = wn.synsets(word)
            if len(syn) > i:
                return str(syn[i].lemmas()[0].name())
            else:
                return ""
        else:
            return ""

class Corpus(object):
    def __init__(self, corpus_path):
        self.train = Dataset(corpus_path, 'train/')
        self.dev = Dataset(corpus_path, 'dev/')
        self.test = Dataset(corpus_path, 'test/')


    @staticmethod
    def get_corpus(corpus_dir, corpus_pkl_path):
        if os.path.exists(corpus_pkl_path):
            with open(corpus_pkl_path, 'rb') as fp:
                corpus= pickle.load(fp)
        else:
            corpus = Corpus(corpus_dir)
            with open(corpus_pkl_path, 'wb') as fp:
                pickle.dump(corpus, fp, -1)

        corpus.print_stats()
        return corpus

    def shuffle_train(self):
        print("##shuffling . . .")
        tokens, words_encoded, labels, emoji, emotion, VAD, intensity= self.train.X, self.train.words_encoded,\
                                                                                     self.train.y, self.train.emoji, self.train.emotion, \
                                                                                     self.train.VAD, self.train.intensity
        z = list(zip(tokens, words_encoded, labels, emoji, emotion, VAD, intensity))
        random.shuffle(z)
        self.train.X, self.train.words_encoded, self.train.y, self.train.emoji, self.train.emotion, self.train.VAD, self.train.intensity=  zip(*z)

    def print_stats(self):
        print("***********************printing status*************************")
        print("Train dataset: {}".format(len(self.train.X)))
        print("Dev dataset: {}".format(len(self.dev.X)))
        print("Test dataset: {}".format(len(self.test.X)))

        print("Train dataset words: {}".format(self.train.X[:2]))
        print("Train dataset labeles: {}".format(self.train.y[:2]))
        print("Train dataset emojifeats: {}".format(self.train.emoji[:2]))

    @staticmethod
    def _get_unique(elems):
        corpus = flatten(elems)
        elems, freqs = zip(*Counter(corpus).most_common())
        return list(elems)

    def get_word_vocab(self):
        num_of_unique_words = self._get_unique(self.train.X_tokenized + self.dev.X_tokenized + self.test.X_tokenized)
        print("[LOG] num_of_unique_words: ", len(num_of_unique_words))
        return num_of_unique_words


class Dataset(object):
    def __init__(self, path, section):
        self.X , self.y, self.X_tokenized= self.read_data(os.path.join(path, section, 'data.csv'))
        self.emoji = self.read_mojiFeatures(os.path.join(config.deepmoji_dir, section, "feat.pkl"), self.X)


    def read_data(self, filename):
        df = pd.read_csv(filename, sep=",")
        list_of_targets = df[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values.tolist()
        list_of_sents = df[["text"]].values.tolist()

        list_of_all_tokenized_sents = []
        list_of_sents_expanded =[]
        for i in list_of_sents:
            i=i[0]
            list_of_sents_expanded.append(i)
            tokenized_doc = tokenizer(i)
            token_lst = [token.text for token in tokenized_doc]
            list_of_all_tokenized_sents.append(token_lst)



        list_of_targets_float =[]
        for l in list_of_targets:
            list_of_targets_float.append([float(y) for y in l])

        return list_of_sents_expanded, list_of_targets_float, list_of_all_tokenized_sents

    def preprocess(self, text):
        text = text.replace('"', "")
        return text


    def read_mojiFeatures(self, file_name_mapped, x):
        mapped = pickle.load(open(file_name_mapped, "rb"))
        feat_lst =[]
        for i in x:
            if i in mapped:
                feat_lst.append(mapped[i])
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! not found!!!!!!!!!!!!!!!!!!!!!!!: ", i)
        print("****read_mojiFeatures - len of features: ", len(feat_lst))
        return feat_lst

