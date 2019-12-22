# built on python3.7

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
import re, pickle
import tensorflow as tf
from tensorflow.keras import backend

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
tokenizer = pickle.load(open("tokenizer.pkl","rb"))

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['sentiment']).values

validation_size = 1500
X_validate = X[-validation_size:]
Y_validate = Y[-validation_size:]
X_test = X[:-validation_size]
Y_test = Y[:-validation_size]

# loading model and validating
batch_size = 32
pos_cnt, neg_cnt, nut_cnt, pos_correct, neg_correct, nut_correct = 0, 0, 0, 0, 0, 0
with graph.as_default():
    model = model_from_json(open("model.json","r").read())
    print (model.summary())
    model.load_weights('model.h5')

    for x in range(len(X_validate)):
        
        result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1)[0]
       
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 2:
                neg_correct += 1
            elif np.argmax(Y_validate[x]) == 1:
                nut_correct += 1
            else:
                pos_correct += 1
           
        if np.argmax(Y_validate[x]) == 2:
            neg_cnt += 1
        elif np.argmax(Y_validate[x]) == 1:
            nut_cnt += 1
        else:
            pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")
print("nut_acc", nut_correct/nut_cnt*100, "%")
    
