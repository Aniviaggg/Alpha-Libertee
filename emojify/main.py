import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow.keras

df = pd.read_csv("data/emoji_uncleaned.csv").sample(n=30)
df.dropna(inplace=True)

X = df["Tweet"].values
y = df["Label"].values

emoji_raw = open('data/us_mapping.txt','r',encoding="utf8")

emojis=[]
for sentence in emoji_raw:
    sentence = sentence.rstrip()
    emojis.append(sentence)

    
emoji_dict={}

for e in emojis:
    idx = int(e.split()[0])
    emoji = e.split()[1]
    emoji_dict[idx] = emoji

from keras_preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open("Model/tweet_tokenizer",'rb'))

def preprocess_text(X):
    max_len=40
    X_seqs = tokenizer.texts_to_sequences(X)
    X_seqs_pd = pad_sequences(X_seqs, truncating="pre", padding="pre", maxlen=max_len)
    return X_seqs_pd

from tensorflow import keras

model = keras.models.load_model("Model/BLSTM.h5", compile=False)
model.summary()

X_sequences = preprocess_text(X)
predictions = np.argmax(model.predict(X_sequences), axis=1)

y_map = [emoji_dict[idx] for idx in y]
predictions = [emoji_dict[idx] for idx in predictions]

results = pd.DataFrame({"Tweet":X})
results["True"] = y_map
results["Predicted"] = predictions

results

import string
import re

from tensorflow import keras
emoji_predict_model = keras.models.load_model("Model/BLSTM.h5", compile=False)


def tweet_clean(tweet):
    tweet = str(tweet).lower()
    rm_mention = re.sub(r'@[A-Za-z0-9]+', '', tweet)                       # remove @mentions
    rm_rt = re.sub(r'RT[/s]+', '', rm_mention)                             # remove RT
    rm_links = re.sub(r'http\S+', '', rm_rt)                               # remove hyperlinks
    rm_links = re.sub(r'https?:\/\/\S+','', rm_links)
    rm_nums = re.sub('[0-9]+', '', rm_links)                               # remove numbers
    rm_punc = [char for char in rm_nums if char not in string.punctuation] # remove punctuations
    rm_punc = ''.join(rm_punc)
    cleaned = rm_punc
    
    return cleaned


def predict_emoji(text, model=emoji_predict_model):
    text = tweet_clean(text)
    X_sequences = preprocess_text([text])
    predictions = np.argmax(model.predict(X_sequences), axis=1)
    emoji_idx = predictions[0]
    emoji = emoji_dict[emoji_idx]
    
    return emoji

text = input("Enter tweet \n\n")

print("\n\n Emojified Tweet \n\n")
print(text+" "+predict_emoji(text))