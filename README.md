# Font_LDL_2020
This is a repository for the ACL 2020 paper: "Let Me Choose: From Verbal Context to Font Selection"

## Requirements
torch==1.1.0<br>
transformers==2.8.0<br>
sklearn==0.21.2<br>
pandas==0.24.2<br>
numpy==1.13.3<br>
nltk==3.4.4<br>
spacy==2.0.13<br>
pickle==4.0<br>

## Data
You can find the Font dataset in the following repository: 
```
https://github.com/RiTUAL-UH/Font-prediction-dataset
```

## Settings required for the Emoji model:
<b> Emoji Model: </b> In this model, we use the Deep-Moji pre-trained model (Felbo et al., 2017) to generate emoji vectors by encoding the text into 2304-dimensional feature vectors. Our implementation is based on the Hugging Face Torch-moji implementation. 
```
https://github.com/huggingface/torchMoji/blob/master/examples/encode_texts.py
```
You can find emoji vectors for the Font dataset in <a href="Font_LDL/DATA/deepmoji_feats/">Font_LDL/DATA/deepmoji_feats/</a>.

## Instructions for running the code
- In <a href="Font_LDL/config.py">config.py</a> select the model and configurations. `base_model` values are `"glove"`, `"bert_seq_classification"`, `"emoji"` and `"NRCfeat"`. (For more information about the details of the models check out the ACL paper)
- Change `train` and `test` to `True` for training and testing respectively. 
- Use `python main.py` for running the model. 

## Citation
```
@inproceedings{shirani2020font,
  title={Let Me Choose: From Verbal Context to Font Selection},
  author={Shirani, Amirreza and Dernoncourt, Franck and Echevarria, Jose and Asente, Paul and Lipka, Nedim and Solorio, Thamar},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```






