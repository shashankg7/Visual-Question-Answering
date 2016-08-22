
from __future__ import print_function
import sys

sys.path.append('../utils')

from vgg16 import VGG16, img_feats
from model_QA import modelQA
from text_handler import parse_QA, gen_vocab, encode_text, encode_ans, vocab_size
import skimage.io as io
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb

IMG_DIR = '/home/shashank/data/VQA/dataset/VQAorg/Images/train2014/'

def vqa_mlp(batch_size=32, epochs=10, max_len=10):
    vgg = VGG16(include_top=True, weights='imagenet')
    gen_vocab()
    model = modelQA(vocab_size, 4096, 300, max_len)
    ques_train, ans_train, img_train, ques_val, ans_val, img_val = parse_QA()
    # Parse all training images and load them into memory
    for epoch in xrange(epochs):
        batch_ind = 1
        Img_feats = []
        ques_feats = []
        labels = []

        for img, ques, ans in zip(img_train, ques_train, ans_train):
            img_path = 'COCO_' + 'train2014' + '_' + str(img).zfill(12) + '.jpg'
            img_feat = img_feats(IMG_DIR + img_path)
            ques_feat = encode_text(ques)
            ans_feat = encode_ans(ans)
            Img_feats.append(img_feat)
            ques_feats.append(ques_feat)
            labels.append(ans_feat)
            if batch_ind % batch_size == 0:
                ques_feats = np.array(ques_feats)
                ques_feats = pad_sequences(ques_feats, 10)
                Img_feats = np.array(Img_feats)
                Img_feats = Img_feats.reshape(batch_size, 4096)

                pdb.set_trace()
                model.train_on_batch([Img_feats, ques_feats], labels)
                batch_ind = 0
                Img_feats = []
                ques_feats = []
                labels = []
            batch_ind += 1

    pdb.set_trace()


def main():
    vqa_mlp()


if __name__ == "__main__":
    main()
