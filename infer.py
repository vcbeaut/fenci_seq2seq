import numpy as np
import pickle
import operator
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jieba
import requests
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Dropout, Concatenate, RepeatVector, Activation, Dot
from keras.layers import concatenate, dot
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import TruncatedNormal
import pydot
import os, re


question = np.load('data/pad_question.npy')
answer = np.load('data/pad_answer.npy')
answer_o = np.load('data/answer_o.npy')
with open('data/vocab_bag.pkl', 'rb') as f:
    words = pickle.load(f)
with open('data/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open('data/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)
vocab_size = len(word_to_index) + 1
maxLen=20

from keras.preprocessing import sequence
def generate_train(batch_size):
    print('\n*********************************generate_train()*********************************')
    steps=0
    question_ = question
    answer_ = answer
    while True:
        batch_answer_o = answer_o[steps:steps+batch_size]
        batch_question = question_[steps:steps+batch_size]
        batch_answer = answer_[steps:steps+batch_size]
        outs = np.zeros([batch_size, maxLen, vocab_size], dtype='float32')
        for pos, i in enumerate(batch_answer_o):
            for pos_, j in enumerate(i):
                if pos_ > 20:
                    print(i)
                outs[pos, pos_, j] = 1 # one-hot
        yield [batch_question, batch_answer], outs
        steps += batch_size
        if steps == 100000:
            steps = 0

def input_question(seq):
    seq = jieba.lcut(seq.strip(), cut_all=False)
    sentence = seq
    try:
        seq = np.array([word_to_index[w] for w in seq])
    except KeyError:
        seq = np.array([36874, 165, 14625])
    seq = sequence.pad_sequences([seq], maxlen=maxLen,
                                          padding='post', truncating='post')
    print(seq)
    return seq, sentence
def decode_greedy(seq, sentence):
    question = seq

    answer = np.zeros((1, 1))
    attention_plot = np.zeros((20, 20))
    answer[0, 0] = word_to_index['BOS']
    i=1
    answer_ = []
    flag = 0
    encoder_lstm_, question_h, question_c = encoder_model.predict(x=question, verbose=1)
#     print(question_h, '\n')
    while flag != 1:
        prediction, prediction_h, prediction_c, attention = decoder_model.predict([
            answer, question_h, question_c, encoder_lstm_
        ])
        attention_weights = attention.reshape(-1, )
        attention_plot[i] = attention_weights
        word_arg = np.argmax(prediction[0, -1, :])#
        answer_.append(index_to_word[word_arg])
        if word_arg == word_to_index['EOS']  or i > 18:
            flag = 1
        answer = np.zeros((1, 1))
        answer[0, 0] = word_arg
        question_h = prediction_h
        question_c = prediction_c
        i += 1
    result = ' '.join(answer_)
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence)]
    plot_attention(attention_plot, sentence, result.split(' '))
    return ' '.join(answer_)
def decode_beamsearch(seq, beam_size):
    question = seq
    encoder_lstm_, question_h, question_c = encoder_model.predict(x=question, verbose=1)
    sequences = [[[word_to_index['BOS']], 1.0, question_h, question_c]]
    answer = np.zeros((1, 1))
    answer[0, 0] = word_to_index['BOS']
    answer_ = ''
    flag = 0
    last_words = [word_to_index['BOS']]
    for i in range(maxLen):
        all_candidates = []
        for j in range(len(sequences)):
            s, score, h, c = sequences[j]
            last_word = s[-1]
            if not isinstance(last_word, int):
                last_word=last_word[-1]
            answer[0, 0] = last_word
            output, h, c, _ = decoder_model.predict([answer, h, c, encoder_lstm_])
            output = output[0, -1]
            for k in range(len(output)):
                candidate = [seq+[k], score*-np.log(output[k]), h, c]
            all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:beam_size]
    answer_ = sequences[0][0]
    print(answer_[0])
    answer_ = [index_to_word[x] for x in answer_[0] if (x!=0)]
    answer_ = ' '.join(answer_)
    return answer_
def plot_attention(attention, sentence, predicted_sentence):
    zhfont = matplotlib.font_manager.FontProperties(fname='data/simkai.ttf')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    attention = [x[::-1] for x in attention]
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 20}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict,fontproperties=zhfont)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict, fontproperties=zhfont)
#     ax.yaxis.set_ticks_position('right') #y轴刻度位置靠右
    plt.show()

def build_model():
    truncatednormal = TruncatedNormal(mean=0.0, stddev=0.05)
    embed_layer = Embedding(input_dim=vocab_size, 
                            output_dim=100, 
                            mask_zero=True,
                            input_length=None,
                            embeddings_initializer= truncatednormal)
    LSTM_encoder = LSTM(512,
                        return_sequences=True,
                        return_state=True,
                        kernel_initializer= 'lecun_uniform',
                        name='encoder_lstm'
                            )
    LSTM_decoder = LSTM(512, 
                        return_sequences=True, 
                        return_state=True, 
                        kernel_initializer= 'lecun_uniform',
                        name='decoder_lstm'
                    )

    #encoder输入 与 decoder输入
    input_question = Input(shape=(None, ), dtype='int32', name='input_question')
    input_answer = Input(shape=(None, ), dtype='int32', name='input_answer')

    input_question_embed = embed_layer(input_question)
    input_answer_embed = embed_layer(input_answer)


    encoder_lstm, question_h, question_c = LSTM_encoder(input_question_embed)

    decoder_lstm, _, _ = LSTM_decoder(input_answer_embed, 
                                    initial_state=[question_h, question_c])

    attention = dot([decoder_lstm, encoder_lstm], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, encoder_lstm], axes=[2,1])
    decoder_combined_context = concatenate([context, decoder_lstm])

    decoder_dense1 = TimeDistributed(Dense(256,activation="tanh"))
    decoder_dense2 = TimeDistributed(Dense(vocab_size,activation="softmax"))
    output = decoder_dense1(decoder_combined_context)
    output = decoder_dense2(output)

    model = Model([input_question, input_answer], output)

    # 构建encoder_model
    encoder_model = Model(input_question, [encoder_lstm, question_h, question_c])

    # 构建decoder_model
    answer_h = Input(shape=(512,))
    answer_c = Input(shape=(512,))
    encoder_lstm = Input(shape=(maxLen,512))
    target, h, c = LSTM_decoder(input_answer_embed, initial_state=[answer_h, answer_c])
    attention = dot([target, encoder_lstm], axes=[2, 2])
    attention_ = Activation('softmax')(attention)
    context = dot([attention_, encoder_lstm], axes=[2,1])
    decoder_combined_context = concatenate([context, target])
    output = decoder_dense1(decoder_combined_context) # equation (5) of the paper
    output = decoder_dense2(output) # equation (6) of the paper
    decoder_model = Model([input_answer, answer_h, answer_c, encoder_lstm], [output, h, c, attention_])

    return model, encoder_model, decoder_model

if __name__ == "__main__":
    model, encoder_model, decoder_model = build_model()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    model.load_weights('models/W-- 51-0.7715-.h5')
    model.summary()
    while True:
        seq = input('Please input question:')
        if seq == 'exit':
            break
        seq, sentence = input_question(seq)
        print(sentence)
        answer = decode_greedy(seq, sentence)
    #     answer=decode_beamsearch(seq, 3)
        print('ANSWER: ', answer)