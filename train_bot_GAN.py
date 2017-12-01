# -*- coding: utf-8 -*-

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence

import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import cPickle
import theano.tensor as T
import os
import h5py
import pandas as pd
import sys
import matplotlib.pyplot as plt
from keras.utils import plot_model

forget = 0  #  (0 to train, 1 to forget)
bidirec = 0
word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50
maxlen_output = 50
number_of_LSTM_layers = 2
decoder_size = 3500
num_subsets = 1
Epochs = 4
BatchSize = 128  #  Check the capacity of your GPU
Patience = 0
dropout = .25
n_test = 3
n_val = 700
learning_rate = 0.000001

vocabulary_file = 'vocabulary_movie'
questions_file = 'Padded_context_generated'
answers_file = 'Padded_answers_generated'
weights_file = 'my_model_weights_bot.h5'
weights_file_discriminator = 'my_model_weights_discriminator.h5'
GLOVE_DIR = './glove.6B/'

# Loading weights of discriminator (frozen in this bot training): 

encoder_matrix = cPickle.load(open('weights_discriminator_layer_3', 'rb'))
print (len(encoder_matrix))
decoder_matrix = cPickle.load(open('weights_discriminator_layer_4', 'rb'))
dense_matrix = cPickle.load(open('weights_discriminator_layer_7', 'rb'))
print('Discriminator model loaded.')


def negative_crossentropy(y_true, y_pred):
    return ( - K.categorical_crossentropy(y_true, y_pred))

def print_result(input):

    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([input, ans_partial])
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = vocabulary[k]
            text = text + w[0] + ' '
    return(text)

# **********************************************************************
# Reading a pre-trained word embedding and addapting to our vocabulary:
# **********************************************************************

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((dictionary_size, word_embedding_size))

# Loading our vocabulary:
vocabulary = cPickle.load(open(vocabulary_file, 'rb'))

# Using the Glove embedding:
i = 0
for word in vocabulary:
    embedding_vector = embeddings_index.get(word[0])
    
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    i += 1


def init_model(bidirec, flag2):

    # *******************************************************************
    # Keras model of the chatbot: 
    # *******************************************************************

    ad = Adam(lr=learning_rate) 

    input_context = Input(shape=(maxlen_input,), dtype='int32', name='input context')
    input_answer = Input(shape=(maxlen_input,), dtype='int32', name='input answer')

    if bidirec == 1:
        LSTM_encoder_bot = Bidirectional(LSTM(sentence_embedding_size, init= 'lecun_uniform', name = 'encoder bot'))
    else:
        LSTM_encoder_bot = LSTM(sentence_embedding_size, init= 'lecun_uniform', name = 'encoder bot')
        
    LSTM_decoder_bot = LSTM(sentence_embedding_size, init= 'lecun_uniform', name = 'decoder bot')
    
    if bidirec == 1:
        LSTM_encoder_discriminator = Bidirectional(LSTM(sentence_embedding_size, init= 'lecun_uniform'), trainable=False, name = 'encoder discriminator')
    else:
        LSTM_encoder_discriminator = LSTM(sentence_embedding_size, init= 'lecun_uniform', trainable=False, name = 'encoder discriminator')
        
    LSTM_decoder_discriminator = LSTM(sentence_embedding_size, init= 'lecun_uniform', trainable=False, name = 'decoder discriminator')
    
    if os.path.isfile(weights_file):
        Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, trainable=False, name = 'shared')
    else:
        Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix], input_length=maxlen_input, trainable=False, name = 'shared')

    word_embedding_context = Shared_Embedding(input_context)
    context_embedding_bot = LSTM_encoder_bot(word_embedding_context)

    word_embedding_answer = Shared_Embedding(input_answer)
    answer_embedding_bot = LSTM_decoder_bot(word_embedding_answer)
    
    context_embedding_discriminator = LSTM_encoder_discriminator(word_embedding_context)

    answer_embedding_discriminator = LSTM_decoder_discriminator(word_embedding_answer)

    merge_layer = merge([context_embedding_bot, answer_embedding_bot], mode='concat', concat_axis=1, name = 'concatenation bot')
    out = Dense(dictionary_size/2, activation="relu", name = 'bot')(merge_layer)
    out = Dense(dictionary_size, activation="softmax", name = 'decision bot')(out)
    loss = merge([context_embedding_discriminator, answer_embedding_discriminator, out], mode='concat', concat_axis=1, name = 'concatenation discriminator')
    loss = Dense(1, activation="sigmoid", trainable=False, name = 'discriminator output')(loss)
   
    model = Model(input=[input_context, input_answer], output = [loss])

    if forget == 0:
        model.compile(loss='mse', optimizer=ad)
    else:
        model.compile(loss='negative_crossentropy', optimizer=ad)
    
    for kk in range (12):
        print('layer %d'%kk)
        print(model.layers[kk])    
         
    plot_model(model, to_file='model_bot_graph.png') 

    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
        
    model.layers[7].set_weights(encoder_matrix)
    model.layers[8].set_weights(decoder_matrix)
    model.layers[11].set_weights(dense_matrix)
        
    return model    

# ************************************************************************
# Loading the data:
# ************************************************************************

q = cPickle.load(open(questions_file, 'rb'))
a = cPickle.load(open(answers_file, 'rb'))
n_exem, n_words = a.shape

qt = q[0:n_test,:]
at = a[0:n_test,:]
q = q[n_test + 1:,:]
a = a[n_test + 1:,:]

print('Number of examples = %d'%(n_exem - n_test))
step = np.around((n_exem - n_test)/num_subsets)
round_exem = step * num_subsets

# *************************************************************************
# Bot training:
# *************************************************************************

x = range(0,Epochs) 
valid_loss = np.zeros(Epochs)
train_loss = np.zeros(Epochs)
max_loss_valid = 100000
patience = 0
m = 0

model = init_model(bidirec, 0)
flag = 0

while (m < Epochs) and (patience <= Patience):
    
    m += 1
    # Loop over training batches due to memory constraints:
    for n in range(0,round_exem,step):
        
        q2 = q[n:n+step]
        s = q2.shape
        count = 0
        for i, sent in enumerate(a[n:n+step]):
            
            l = np.where(sent==3)  #  the position od the symbol EOS
            limit = l[0][0]
            count += limit + 1
          
        Q = np.zeros((count,maxlen_input))
        A = np.zeros((count,maxlen_input))
        Y = np.ones(count)
        
        # Loop over the training examples:
        count = 0
        for i, sent in enumerate(a[n:n+step]):
            ans_partial = np.zeros((1,maxlen_input))
            
            # Loop over the positions of the current target output (the current output sequence):
            l = np.where(sent==3)  #  the position of the symbol EOS
            limit = l[0][0]

            for k in range(1,limit+1):

                # preparing the partial answer to input:

                ans_partial[0,-k:] = sent[0:k]

                # training the model for one epoch using teacher forcing:
                
                Q[count, :] = q2[i:i+1] 
                A[count, :] = ans_partial 
                count += 1
        
        print('Training epoch: %d, training examples: %d - %d'%(m,n, n + step))        
        
        model.fit([Q[n_val:,:], A[n_val:,:]], Y[n_val:], batch_size=BatchSize, epochs=1, validation_split=0.0)
        model.save_weights(weights_file, overwrite=True)

         
        test_input = qt[1:2]
        print(print_result(test_input))
        train_input = q[1:2]
        print(print_result(train_input))        
        
    model.save_weights(weights_file, overwrite=True)
    