# -*- coding: utf-8 -*-

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import cPickle
import theano
import os.path
import sys
import nltk
import re
import time
from keras.utils import plot_model

word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50
maxlen_input_consist = 40
dense_size = 1500

questions_file = 'context_simple'
vocabulary_file = 'vocabulary_movie'
weights_file_baseline = 'my_model_weights.h5'
weights_file = 'my_model_weights_bot.h5'
unknown_token = 'something'
file_generated_context = 'generated_context'
file_generated_answer = 'generated_answer'
name_of_computer = 'john'
name = 'john'
depth_of_thinking = 2
threshold_max = 0.65
threshold_min = 0.50
pressure = 4.
bidirec = 0

print('Starting the model...')

# *******************************************************************
# Keras model of the chatbot + discriminator: 
# *******************************************************************

ad = Adam(lr=0.00005) 

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


if os.path.isfile(weights_file):
    print('loading the bot weights...')
    model.load_weights(weights_file)

# *******************************************************************
# Only the model of the chatbot:
# *******************************************************************

input_context2 = Input(shape=(maxlen_input,), dtype='int32', name='input context')
input_answer2 = Input(shape=(maxlen_input,), dtype='int32', name='input answer')

if bidirec == 1:
    LSTM_encoder_bot2 = Bidirectional(LSTM(sentence_embedding_size, init= 'lecun_uniform', weights=model.layers[3].get_weights(), name = 'encoder bot'))
else:
    LSTM_encoder_bot2 = LSTM(sentence_embedding_size, init= 'lecun_uniform', weights=model.layers[3].get_weights(), name = 'encoder bot')
    
LSTM_decoder_bot2 = LSTM(sentence_embedding_size, init= 'lecun_uniform', weights=model.layers[4].get_weights(), name = 'decoder bot')

Shared_Embedding2 = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, trainable=False, weights=model.layers[2].get_weights(), name = 'shared')

word_embedding_context2 = Shared_Embedding2(input_context2)
context_embedding_bot2 = LSTM_encoder_bot2(word_embedding_context2)

word_embedding_answer2 = Shared_Embedding2(input_answer2)
answer_embedding_bot2 = LSTM_decoder_bot2(word_embedding_answer2)
merge_layer2 = merge([context_embedding_bot2, answer_embedding_bot2], mode='concat', concat_axis=1, name = 'concatenation bot')
out2 = Dense(dictionary_size/2, activation="relu", weights=model.layers[6].get_weights(), name = 'bot')(merge_layer2)
out2 = Dense(dictionary_size, activation="softmax", weights=model.layers[9].get_weights(), name = 'decision bot')(out2)

model2 = Model(input=[input_context2, input_answer2], output = [out2])

model2.save_weights(weights_file_baseline, overwrite=True)
