# Font_LDL_2020
This is a repository for the ACL 2020 paper: "Let Me Choose: From Verbal Context to Font Selection"

## Requirements
python
torch
transformers
sklearn
pandas
numpy
NLTK
spacy
pickle

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
You can find emoji vectors for the Font dataset <a href="https://drive.google.com/drive/folders/1BRMWfWk9P7Uc3b8r9xlm6lbu0f22nqQ6?usp=sharing">here</a>. 


