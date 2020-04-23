import torch

torch_random_seed =123
np_random_seed =123
random_random_seed = 123

gpu_number =5

base_model = "bert_seq_classification" #values: "glove", "bert_seq_classification", "emoji", "NRCfeat"
version_num = "V1"

version = "{}_{}".format(base_model, version_num)

if_shuffle = True
##############################################################
epoch = 800
train = True
test = True
learning_rate = 0.00001
batch_size = 16
############################################################## Features
use_NRCemotion = False
use_NRCintensity = False
use_NRCvad = False
use_useDeepMojiFeat = False
if base_model == "emoji":
    use_useDeepMojiFeat = True
if base_model == "NRCfeat":
    use_NRCemotion = True
##############################################################
corpus_pkl  = "DATA/pkl/corpus_V5.pkl"
corpus_dir = "DATA/font"
encoder_pkl = "DATA/pkl/encoder_V5.pkl"
deepmoji_dir = "DATA/deepmoji_feats"
##############################################################
output_dir_path = "models_checkpoints/"+version+"/"
score_output = "outputs/"+version+"/"
emotion_path ='DATA/emotion_lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
VAD_path = 'DATA/emotion_lexicon/NRC-VAD-Lexicon.txt'
intensity_path = 'DATA/emotion_lexicon/NRC-AffectIntensity-Lexicon.txt'
##############################################################
if not torch.cuda.is_available():
    print("[LOG] running on CPU")
    emb_path ='EMBEDDINGS/glove.6B/glove.6B.100d.txt'
else:
    print("[LOG] running on GPU")
    emb_path = '../../../embedding/glove.6B.100d.txt'
##############################################################