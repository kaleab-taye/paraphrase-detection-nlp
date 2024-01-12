# Importing packages
import pandas as pd
import numpy as np
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import model
import pickle
import math

sp = spacy.load('en_core_web_sm')

path = '/home/passion/Desktop/Everything/paraphrase-detection-nlp/Model and Server/server/resources'

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

tokenizer = ""
embeddings_index = ""
with open(path+'/tokenizer.pkl', 'rb') as inp1:
    tokenizer = pickle.load(inp1)
with open(path+'/embeddings_index.pkl', 'rb') as inp2:
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
        
model = model.create_model(input_shape,
                      embedding_dim,
                     embedding_matrix, vocab_size,
                      max_len, trainable_embeddings, dropout,
                      lstm_hidden_units, attention_channels, pool_size,
                      fc_hidden_units)

model.load_weights(path+"/model_paraprase_detection_pad.h5")  # Replace with the actual path

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

new_test_data_dict = {
    'question1': ["How does photosynthesis work?", "What are the benefits of exercise?", "Python vs Java"],
    'question2': ["what is the working mechanism of photosynthesis.", "where is the closest supermarket around here?", "Comparison between Python and Java"],
    'is_duplicate': [1, 0, 1]  # The true labels (1 for duplicate, 0 for non-duplicate)
}

test = getSimilarity(new_test_data_dict["question1"][0],new_test_data_dict["question2"][0])
print(test)
test = getSimilarity(new_test_data_dict["question1"][1],new_test_data_dict["question2"][1])
print(test)
test = getSimilarity(new_test_data_dict["question1"][2],new_test_data_dict["question2"][2])
print(test)