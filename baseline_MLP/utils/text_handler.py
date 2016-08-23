# 1st Baseline with Image + Caption model merged by simple MLP on top
from vqa import VQA
import numpy as np
from string import punctuation
from collections import defaultdict
import os
import pdb
import json


DIR = "/home/shashank/data/VQA/dataset/VQAorg"
ANNOTATION_TRAIN_PATH = '%s/Annotations/mscoco_train2014_annotations.json'%(DIR)
ANNOTATION_VAL_PATH = '%s/Annotations/mscoco_val2014_annotations.json'%(DIR)
QUES_TRAIN_PATH = '%s/Questions/MultipleChoice_mscoco_train2014_questions.json'%(DIR)
QUES_VAL_PATH = '%s/Questions/MultipleChoice_mscoco_val2014_questions.json'%(DIR)
GLOVE_PATH = '%s/WordEmbeddings/glove.6B.100d.txt'%(DIR)
vqa_train = VQA(ANNOTATION_TRAIN_PATH, QUES_TRAIN_PATH)
vqa_val = VQA(ANNOTATION_VAL_PATH, QUES_VAL_PATH)
vocab = {}
vocab_size = 0
embedding_dim = 100

def filter_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    return text


def parse_QA(ques_type='yes/no'):
    """
    Returns a list of all questions given the question type
    """
    if ques_type == 'yes/no':
        annIds_train = vqa_train.getQuesIds(ansTypes='yes/no')
        anns_train = vqa_train.loadQA(annIds_train)
        annIds_val = vqa_val.getQuesIds(ansTypes='yes/no')
        anns_val = vqa_val.loadQA(annIds_val)
        # Get questions correspoding to the question ids
        ques_train = []
        answers_train = []
        ques_val = []
        answers_val =[]
        imgIds_train = []
        imgIds_val = []

        for id in annIds_train:
            ques_train.append(vqa_train.qqa[id]['question'])
        for ann in anns_train:
            answers_train.append(ann['multiple_choice_answer'])
            imgIds_train.append(ann['image_id'])

        for id in annIds_val:
            ques_val.append(vqa_val.qqa[id]['question'])
        for ann in anns_val:
            answers_val.append(ann['multiple_choice_answer'])
            imgIds_val.append(ann['image_id'])
        return ques_train, answers_train, imgIds_train, ques_val, answers_val\
            ,imgIds_val

    elif ques_type == 'how many':
        return NotImplementedError

def gen_vocab():
    global vocab_size
    ques_train, ans_train, _, ques_val, ans_val, _ = parse_QA()
    #pdb.set_trace()
    word_idx = 0
    for ques in ques_train:
        ques = filter_text(ques)
        for word in ques.split():
            if word not in vocab:
                vocab[word] = word_idx
                word_idx += 1
    vocab['UNK'] = word_idx
    vocab_size = word_idx

def get_vocab_size():
    global vocab_size
    return vocab_size

def encode_text(text):
    text = filter_text(text)
    res = []
    for word in text.split():
        if word in vocab:
            res.append(vocab[word])
        else:
            res.append(vocab['UNK'])
    return res

def encode_ans(ans):
    if ans == 'yes':
        return 1
    else:
        return 0

def load_embeddings():
    embeddings_index = {}
    f = open(GLOVE_PATH, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for key, val in vocab.items():
        embedding = embeddings_index.get(key)
        if embedding is not None:
            embedding_matrix[val] = embedding
    pdb.set_trace()
    return embedding_matrix

if __name__ == "__main__":
    gen_vocab()
    load_embeddings()
    pdb.set_trace()
