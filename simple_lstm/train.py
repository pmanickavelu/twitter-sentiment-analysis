# built on python3.7

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import re, pickle
import tensorflow as tf

graph = tf.Graph()

# lodaing data
data = pd.read_csv('../data/Sentiment.csv')

# cleaning the data
data = data[['text','sentiment']]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
 

# data prepration/feature extraction/feature building
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['sentiment']).values



# model building/model defenitions and training
embed_dim = 128
lstm_out = 196
batch_size = 32
epochs = 7
with graph.as_default():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.fit(X, Y, epochs = epochs, batch_size=batch_size)
# saveing the model
    model.save_weights('model.h5')
with open("model.json","w") as f:
    f.write(model.to_json())
pickle.dump(tokenizer,open("tokenizer.pkl","wb"))
pickle.dump(X.shape[1],open("max_text_length.pkl","wb"))

    
