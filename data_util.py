import numpy as np
import pickle
import operator
import pandas as pd
import jieba
from language.langconv import *
from keras.preprocessing import sequence

word_to_index = {}
index_to_word = {}
vocab_bag = None
data_path = 'data/qingyun.tsv'
num_samples = 100000
raw_maxLen = 18
pad_maxLen = 20
def Traditional2Simplified(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def is_all_chinese(strs):
    for chart in strs:
        if chart < u'\u4e00' or chart > u'\u9fff':
            return False
    return True

def get_raw_data():
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        lines = lines[:-1]
        lines = lines[:min(num_samples,len(lines)-1)]
    question = []
    answer = []
    for pos, line in enumerate(lines):
        if '\t' not in line:
            print(line)
        line = line.split('\t')
        q = line[0].strip()
        a = line[1].strip()
        question.append(' '.join(jieba.lcut(Traditional2Simplified(q).strip(), cut_all=False)))
        answer.append(' '.join(jieba.lcut(Traditional2Simplified(a).strip(), cut_all=False)))
    return question, answer
    
    
    
def get_qa_data(question, answer):
    character = set()
    for seq in question + answer:
        word_list = seq.split(' ')
        for word in word_list:
            if not is_all_chinese(word):
                character.add(word)
    def is_pure_english(keyword):  
        return all(ord(c) < 128 for c in keyword)
    character=list(character)
    stop_words = set()
    for pos, word in enumerate(character):
        if not is_pure_english(word):
            stop_words.add(word)

    for pos, seq in enumerate(question):
        seq_list = seq.split(' ')
        for epoch in range(3):#这里需要多次epoch，因为可能会有多个停用词，有pos位置跳过
            for pos_, word in enumerate(seq_list):
                if word in stop_words:
                    seq_list.pop(pos_)
        if len(seq_list) > raw_maxLen:
            seq_list = seq_list[:raw_maxLen]
        question[pos] = ' '.join(seq_list)
    for pos, seq in enumerate(answer):
        seq_list = seq.split(' ')
        for epoch in range(3):
            for pos_, word in enumerate(seq_list):
                if word in stop_words:
                    seq_list.pop(pos_)
        if len(seq_list) > raw_maxLen:
            seq_list = seq_list[:raw_maxLen]
        answer[pos] = ' '.join(seq_list)
    return question, answer

def get_voc_dict_with_vocbag(question, answer):
    global word_to_index, index_to_word, vocab_bag
    
    counts = {}
    BE = ['BOS', 'EOS']
    for word_list in question + answer + BE:
        for word in word_list.split(' '):
            counts[word] = counts.get(word, 0) + 1
    
    for pos, i in enumerate(counts.keys()):
        word_to_index[i] = pos
        
    
    for pos, i in enumerate(counts.keys()):
        index_to_word[pos] = i
 
    vocab_bag =list(word_to_index.keys())


def get_qa_vec(question, answer):
    global word_to_index,index_to_word
    answer_a = ['BOS ' + i + ' EOS' for i in answer]
    answer_b = [i + ' EOS' for i in answer]
    question = np.array([[word_to_index[w] for w in i.split(' ')] for i in question])
    answer_a = np.array([[word_to_index[w] for w in i.split(' ')] for i in answer_a])
    answer_b = np.array([[word_to_index[w] for w in i.split(' ')] for i in answer_b])
    for i, j in word_to_index.items():
        word_to_index[i] = j + 1
    
    for key, value in word_to_index.items():
        index_to_word[value] = key
    pad_question = question
    pad_answer_a = answer_a
    pad_answer_b = answer_b
    
    for pos, i in enumerate(pad_question):
        for pos_, j in enumerate(i):
            i[pos_] = j + 1
        if(len(i) > pad_maxLen):
            pad_question[pos] = i[:pad_maxLen]
        
    for pos, i in enumerate(pad_answer_a):
        for pos_, j in enumerate(i):
            i[pos_] = j + 1
        if(len(i) > pad_maxLen):
            pad_answer_a[pos] = i[:pad_maxLen]
    for pos, i in enumerate(pad_answer_b):
        for pos_, j in enumerate(i):
            i[pos_] = j + 1
        if(len(i) > pad_maxLen):
            pad_answer_b[pos] = i[:pad_maxLen]

    return question, answer_a, answer_b


if __name__ == "__main__":
    raw_q, raw_a = get_raw_data()
    question, answer = get_qa_data(raw_q, raw_a)
    print(question[:10])
    print(answer[:10])
    get_voc_dict_with_vocbag(question, answer)
    encoder_input_vec, decoder_input_vec, decoder_target_vec = get_qa_vec(question,answer)
    # padding input对齐
    pad_question = sequence.pad_sequences(encoder_input_vec, maxlen=pad_maxLen,
                                      dtype='int32', padding='post', 
                                       truncating='post')
    pad_answer = sequence.pad_sequences(decoder_input_vec, maxlen=pad_maxLen,
                                    dtype='int32', padding='post',
                                    truncating='post')
    with open('data/word_to_index.pkl', 'wb') as f:
        pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)
    with open('data/index_to_word.pkl', 'wb') as f:
        pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
    with open('data/vocab_bag.pkl','wb') as f:
        pickle.dump(vocab_bag,f,pickle.HIGHEST_PROTOCOL)

    np.save('data/pad_question.npy', pad_question)
    np.save('data/pad_answer.npy', pad_answer)
    np.save('data/answer_o.npy', decoder_target_vec)