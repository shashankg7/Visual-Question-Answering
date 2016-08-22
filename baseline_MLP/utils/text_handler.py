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
vqa_train = VQA(ANNOTATION_TRAIN_PATH, QUES_TRAIN_PATH)
vqa_val = VQA(ANNOTATION_VAL_PATH, QUES_VAL_PATH)
vocab = {}
vocab_size = 0


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
    ques_train, ans_train, _, ques_val, ans_val, _ = parse_QA()
    word_idx = 0
    for ques in ques_train:
        ques = filter_text(ques)
        for word in ques.split():
            if word not in vocab:
                vocab[word] = word_idx
                word_idx += 1
    vocab['UNK'] = word_idx
    vocab_size = word_idx

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


if __name__ == "__main__":
    gen_vocab()
    pdb.set_trace()
