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
import pandas as pd
import sys
import matplotlib.pyplot as plt
import nltk
from keras.utils import plot_model

bidirec = 0
word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50
maxlen_output = 50
number_of_LSTM_layers = 2
decoder_size = 3500
num_subsets = 1
Epochs = 15
BatchSize = 128  #  Check the capacity of your GPU
Patience = 10
dropout = .25
n_test = 3
n_val = 700
learning_rate = 0.000001


vocabulary_file = 'vocabulary_movie'
weights_file = 'my_model_weights_discriminator.h5'
unknown_token = 'something'

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


# Loading our vocabulary:
vocabulary = cPickle.load(open(vocabulary_file, 'rb'))

def init_model(bidirec):

    # *******************************************************************
    # Keras model of the discriminator: 
    # *******************************************************************

    ad = Adam(lr=learning_rate) 

    input_context = Input(shape=(maxlen_input,), dtype='int32', name='input context')
    input_answer = Input(shape=(maxlen_input,), dtype='int32', name='input answer')
    input_current_token = Input(shape=(dictionary_size,), name='input_current_token')

    if bidirec == 1:
        LSTM_encoder_discriminator = Bidirectional(LSTM(sentence_embedding_size, init= 'lecun_uniform'), name = 'encoder discriminator')
    else:
        LSTM_encoder_discriminator = LSTM(sentence_embedding_size, init= 'lecun_uniform', name = 'encoder discriminator')
        
    LSTM_decoder_discriminator = LSTM(sentence_embedding_size, init= 'lecun_uniform', name = 'decoder discriminator')
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, trainable=False, name = 'shared')
    word_embedding_context = Shared_Embedding(input_context)
    word_embedding_answer = Shared_Embedding(input_answer)
    context_embedding_discriminator = LSTM_encoder_discriminator(word_embedding_context)
    answer_embedding_discriminator = LSTM_decoder_discriminator(word_embedding_answer)
    loss = merge([context_embedding_discriminator, answer_embedding_discriminator, input_current_token], mode='concat', concat_axis=1, name = 'concatenation discriminator')
    loss = Dense(1, activation="sigmoid", name = 'discriminator output')(loss)

    model = Model(input=[input_context, input_answer, input_current_token], output = [loss])

    model.compile(loss='binary_crossentropy', optimizer=ad)
        
    plot_model(model, to_file='model_discriminator_graph.png') 

    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
        
    return model    

def preprocess(raw_word):
    
    l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
    l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

    raw_word = raw_word.lower()

    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')
    
    for j in range(30):
        raw_word = raw_word.replace('. .', '')
        raw_word = raw_word.replace('.  .', '')
        raw_word = raw_word.replace('..', '')
       
    for j in range(5):
        raw_word = raw_word.replace('  ', ' ')
        
    if raw_word[-1] <>  '!' and raw_word[-1] <> '?' and raw_word[-1] <> '.' and raw_word[-2:] <>  '! ' and raw_word[-2:] <> '? ' and raw_word[-2:] <> '. ':
        raw_word = raw_word + ' .'
    
    if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
        raw_word = 'what ?'
    
    if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
        raw_word = 'i do not want to talk about it .'
    
    raw_word = 'BOS' + raw_word + ' EOS'  
    return raw_word

def tokenize(sentences):

    # Tokenizing the sentences into words:
    tokenized_sentences = nltk.word_tokenize(sentences.decode('utf-8'))
    index_to_word = [x[0] for x in vocabulary]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]
    X = np.asarray([word_to_index[w] for w in tokenized_sentences])
    s = X.size
    Q = np.zeros((1,maxlen_input))
    if s < (maxlen_input + 1):
        Q[0,- s:] = X
    else:
        Q[0,:] = X[- maxlen_input:]
    
    return Q, s
    
def run_discriminator(qr, ar):
    qr  = preprocess(qr)
    q, sq = tokenize(qr)
    ar  = preprocess(ar)
    a, sa = tokenize(ar)

    # *************************************************************************
    # running discriminator:
    # *************************************************************************

    x = range(0,Epochs) 
    p = 1
    m = 0
    model = init_model(bidirec)
    s = q.shape
    count = 0
 
    for i, sent in enumerate(a):
        l = np.where(sent==3)  #  the position od the symbol EOS
        limit = l[0][0]
        count += limit + 1

    Q = np.zeros((count,maxlen_input))
    A = np.zeros((count,maxlen_input))
    Y = np.zeros((count,dictionary_size))

    # Loop over the training examples:
    count = 0
    for i, sent in enumerate(a):
        ans_partial = np.zeros((1,maxlen_input))
        
        # Loop over the positions of the current target output (the current output sequence):
        l = np.where(sent==3)  #  the position of the symbol EOS
        limit = l[0][0]

        for k in range(1,limit+1):
            # Mapping the target output (the next output word) for one-hot codding:
            y = np.zeros((1, dictionary_size))
            y[0, int(sent[k])] = 1

            # preparing the partial answer to input:
            ans_partial[0,-k:] = sent[0:k]

            # training the model for one epoch using teacher forcing:
            Q[count, :] = q[i:i+1] 
            A[count, :] = ans_partial 
            Y[count, :] = y
            count += 1

    p = model.predict([ Q, A, Y])
    p = p[-sa:]
    P = np.sum(np.log(p))/sa
    
    return P

qr = ''
while qr <> 'exit .':
    qr = raw_input('context: ')
    ar = raw_input('answer TF: ')
    ar2 = raw_input('answer GAN: ')
    p1 = run_discriminator(qr, ar)
    p2 = run_discriminator(qr, ar2)
    print p1
    print p2
    if p1>p2:
        best = ar
    else:
        best = ar2
    init = ''
    if abs(p1-p2)/max([p1, p2]) < 0.05:
        init = 'It is almost a tie, but '

    print('the best answer was ' + best)
