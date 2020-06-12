
# Font LDL 2020
This is a repository for our ACL 2020 paper [Let Me Choose: From Verbal Context to Font Selection](https://arxiv.org/abs/2005.01151).


## Requirements

python==3.6.9 and `pip install -r requirements.txt`

## Data
You can find the Font dataset in the following repository: https://github.com/RiTUAL-UH/Font-prediction-dataset

## Settings required for the Emoji model:
<b> Emoji Model: </b> In this model, we use the Deep-Moji pre-trained model (Felbo et al., 2017) to generate emoji vectors by encoding the text into 2304-dimensional feature vectors. Our implementation is based on the [Hugging Face Torch-moji implementation](https://github.com/huggingface/torchMoji/blob/master/examples/encode_texts.py). 
You can find emoji vectors for the Font dataset <a href="Font_LDL/DATA/deepmoji_feats/">here</a>. 

## Instructions for running the code
- `pip install -r requirements.txt`
- `python -m nltk.downloader wordnet`
- Download http://nlp.stanford.edu/data/glove.6B.zip and unzip `glove.6B.100d.txt` (part of `glove.6B.zip`) to `EMBEDDINGS/glove.6B/glove.6B.100d.txt`.
- Download http://sentiment.nrc.ca/lexicons-for-research/NRC-Sentiment-Emotion-Lexicons.zip, then unzip `NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/*` and `NRC-Sentiment-Emotion-Lexicons/NRC-VAD-Lexicon/*` to `DATA/emotion_lexicon`.
- Download http://saifmohammad.com/WebDocs/NRC-AffectIntensity-Lexicon.txt and copy `NRC-AffectIntensity-Lexicon.txt` to `DATA/emotion_lexicon`.
- In <a href="Font_LDL/config.py">config.py</a> select the model and configurations. `base_model` values are `"glove"`, `"bert_seq_classification"`, `"emoji"` and `"NRCfeat"`. (For more information about the details of the models check out the ACL paper)
- Change `train` and `test` to `True` for training and testing respectively. 
- Use `python main.py` for running the model. 

## Citation

If you use this code in your work, please cite our [paper](https://arxiv.org/abs/2005.01151) as follows:

```
@inproceedings{shirani2020font,
  title={Let Me Choose: From Verbal Context to Font Selection},
  author={Shirani, Amirreza and Dernoncourt, Franck and Echevarria, Jose and Asente, Paul and Lipka, Nedim and Solorio, Thamar},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
