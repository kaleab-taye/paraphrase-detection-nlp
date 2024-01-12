# Importing packages
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import string
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation,MaxPooling2D,Bidirectional,Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import spacy
sp = spacy.load('en_core_web_sm')


class GatedRelevanceNetwork(Layer):
    def __init__(self, output_dim,
            weights_initializer="glorot_uniform",
            bias_initializer="zeros", **kwargs):
        self.output_dim = output_dim
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        super(GatedRelevanceNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size, len1, emb_dim = input_shape[0]
        _, len2, _ = input_shape[1]
        # Weights initialization
        # Bilinear Tensor Product weights
        self.Wb = self.add_weight(name='weights_btp',
                                  shape=(self.output_dim, emb_dim, emb_dim),
                                  initializer=self.weights_initializer,
                                  trainable=True)

        # Single Layer Network weights
        self.Wd = self.add_weight(name='weights_sln',
                                  shape=(2*emb_dim, self.output_dim),
                                  initializer=self.weights_initializer,
                                  trainable=True)

        # Gate weights
        self.Wg = self.add_weight(name='weights_gate',
                                  shape=(2*emb_dim, self.output_dim),
                                  initializer=self.weights_initializer,
                                  trainable=True)

        # Gate bias
        self.bg = self.add_weight(name='bias_gate',
                                  shape=(self.output_dim,),
                                  initializer=self.bias_initializer,
                                  trainable=True)

        # General bias
        self.b = self.add_weight(name='bias',
                                 shape=(self.output_dim,),
                                 initializer=self.bias_initializer,
                                 trainable=True)

        # Channel weights
        self.u = self.add_weight(name="channel_weights",
                                 shape=(self.output_dim, 1),
                                 initializer=self.weights_initializer,
                                 trainable=True)

        super(GatedRelevanceNetwork, self).build(input_shape)

    def call(self, x):
        e1 = x[0]
        e2 = x[1]

        batch_size = K.shape(e1)[0]
        # Usually len1 = len2 = max_seq_length
        _, len1, emb_dim = K.int_shape(e1)
        _, len2, _ = K.int_shape(e2)

        # Repeating the matrices to generate all the combinations
        ne1 = K.reshape(K.repeat_elements(K.expand_dims(e1, axis=2), len2, axis=2),
                       (batch_size, len1*len2, emb_dim))
        ne2 = K.reshape(K.repeat_elements(K.expand_dims(e2, axis=1), len1, axis=1),
                       (batch_size, len1*len2, emb_dim))

        # Repeating the second matrix to use in Bilinear Tensor Product
        ne2_k = K.repeat_elements(K.expand_dims(ne2, axis=-1), self.output_dim, axis=-1)

        # Bilinear tensor product
        btp = K.sum(ne2_k * K.permute_dimensions(K.dot(ne1, self.Wb), (0,1,3,2)), axis=2)
        btp = K.reshape(btp, (batch_size, len1, len2, self.output_dim))

        # Concatenating inputs to apply Single Layer Network
        e = K.concatenate([ne1, ne2], axis=-1)

        # Single Layer Network
        #sln = K.relu(K.dot(e, self.Wd))
        sln = K.tanh(K.dot(e, self.Wd))
        sln = K.reshape(sln, (batch_size, len1, len2, self.output_dim))

        # Gate
        g = K.sigmoid(K.dot(e, self.Wg) + self.bg)
        g = K.reshape(g, (batch_size, len1, len2, self.output_dim))

        # Gated Relevance Network
        #s = K.reshape(K.dot(g*btp + (1-g)*sln + self.b, self.u), (batch_size, len1, len2))
        s = K.dot(g*btp + (1-g)*sln + self.b, self.u)

        return s

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        return (shape1[0], shape1[1], shape2[1], 1)

table = str.maketrans('', '', string.punctuation)
def clean_question(text):
    doc = sp(text)
    # tokenize
    # text = text.split()
    # Lemmatization
    text = [token.lemma_ for token in doc]
    # convert to lower case
    text = [word.lower() for word in text]
    # remove punctuation from each token
    text = [w.translate(table) for w in text]
    # remove hanging 's' and 'a'
    text = [word for word in text if len(word)>1]
    # remove tokens with numbers in them
    text = [word for word in text if word.isalpha()]
    # store as string
    return ' '.join(text)

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

tokenizer = ""
embeddings_index = ""
with open('/kaggle/input/ssssss/tokenizer.pkl', 'rb') as inp1:
    tokenizer = pickle.load(inp1)
with open('/kaggle/input/eeeeee2/embeddings_index.pkl', 'rb') as inp2:
    embeddings_index = pickle.load(inp2)
max_len = 25
embedding_dim = 200
dropout = 0.5
trainable_embeddings = False
lstm_hidden_units = 50
attention_channels = 2
pool_size = 3
fc_hidden_units = 128
input_shape = (max_len,)
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
        
model = create_model(input_shape,
                      embedding_dim,
                     embedding_matrix, vocab_size,
                      max_len, trainable_embeddings, dropout,
                      lstm_hidden_units, attention_channels, pool_size,
                      fc_hidden_units)

model.load_weights("/kaggle/input/bbbbbb/model_paraprase_detection_pad.h5")  # Replace with the actual path

# Assume you have a new set of test data in a dictionary format
new_test_data_dict = {
    'question1': ["How does photosynthesis work?", "What are the benefits of exercise?", "Python vs Java"],
    'question2': ["what is the working mechanism of photosynthesis.", "where is the closest supermarket around here?", "Comparison between Python and Java"],
    'is_duplicate': [1, 0, 1]  # The true labels (1 for duplicate, 0 for non-duplicate)
}

# Create a DataFrame
new_test_data = pd.DataFrame(new_test_data_dict)

# Preprocess the test data (similar to what you did for training data)
new_test_data["question1"] = new_test_data["question1"].apply(lambda x: clean_question(x))
new_test_data["question2"] = new_test_data["question2"].apply(lambda x: clean_question(x))

# Tokenize and pad sequences
q1_texts_seq_test = tokenizer.texts_to_sequences(new_test_data["question1"].values)
q2_texts_seq_test = tokenizer.texts_to_sequences(new_test_data["question2"].values)

q1_texts_seq_test = pad_sequences(q1_texts_seq_test, maxlen=max_len)
q2_texts_seq_test = pad_sequences(q2_texts_seq_test, maxlen=max_len)

# Assuming 'is_duplicate' is the column with true labels
true_labels = new_test_data["is_duplicate"].values

# Make predictions
predictions = model.predict([q1_texts_seq_test, q2_texts_seq_test])

# Assuming your model outputs probabilities for each class (binary classification)
# If you used softmax activation in the output layer, you can use argmax to get the predicted class
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate the predictions
# ...

# Evaluate the predictions
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

import math
def getSimilarity(s1, s2):
    test_data = {
        's1': [s1],
        's2': [s2],
    }
    test_data_frame = pd.DataFrame(test_data)
    test_data_frame["s1"] = test_data_frame["s1"].apply(lambda x: clean_question(x))
    test_data_frame["s2"] = test_data_frame["s2"].apply(lambda x: clean_question(x))

    s1_texts_seq_test = tokenizer.texts_to_sequences(test_data_frame["s1"].values)
    s2_texts_seq_test = tokenizer.texts_to_sequences(test_data_frame["s2"].values)

    s1_texts_seq_test = pad_sequences(s1_texts_seq_test, maxlen=max_len)
    s2_texts_seq_test = pad_sequences(s2_texts_seq_test, maxlen=max_len)

    assessment = model.predict([s1_texts_seq_test, s2_texts_seq_test])
    assessmentP = assessment[:, 1] * 100
    similarity= assessmentP.astype(float)[0]
    similarity = math.trunc(similarity*100)/100
    lable = "Not Paraphrase"
    if similarity > 70:
        lable = "Paraphrase"
    return similarity, lable

