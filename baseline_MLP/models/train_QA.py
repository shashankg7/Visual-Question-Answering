
from __future__ import print_function
import sys

sys.path.append('../utils')

from vgg16 import VGG16, img_feats
from model_QA import modelQA
from text_handler import parse_QA, gen_vocab, encode_text, encode_ans, get_vocab_size, load_embeddings
import skimage.io as io
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import pdb

IMG_DIR = '../../VQAorg/Images/'

def vqa_mlp(batch_size=32, epochs=4, max_len=10):
    vgg = VGG16(include_top=True, weights='imagenet')
    gen_vocab()
    embeddings = load_embeddings()
    vocab_size = get_vocab_size()
    model = modelQA(vocab_size + 1, 4096, 100, max_len, embeddings)


    ques_train, ans_train, img_train, ques_val, ans_val, img_val = parse_QA()
    pdb.set_trace()
    # Parse all training images and load them into memory
    for epoch in xrange(epochs):
        Img_feats = []
        ques_feats = []
        labels = []
        # Indices over training data
        train_ind = range(len(ques_train))
        random.shuffle(train_ind)
        batch_ind = 0
        n = len(ques_train)
        # iterate over training exampled in shuffled indices order
        for i in xrange(0, (n/batch_size) + 1):
            batch_index = train_ind[batch_ind * batch_size: (batch_ind + 1) * batch_size]
            img_batch = img_train[batch_ind * batch_size: (batch_ind + 1) * batch_size]
            ques_batch = ques_train[batch_ind * batch_size: (batch_ind + 1) * batch_size]
            ans_batch = ans_train[batch_ind * batch_size: (batch_ind + 1) * batch_size]

            for img in img_batch:
                img_path = 'COCO_' + 'train2014' + '_' + str(img).zfill(12) + '.jpg'
                img_feat = img_feats(IMG_DIR +'train2014/' +  img_path)
                Img_feats.append(img_feat)

            for ques in ques_batch:
                ques_feat = encode_text(ques)
                ques_feats.append(ques_feat)

            for ans in ans_batch:
                ans_feat = encode_ans(ans)
                labels.append(ans_feat)

            ques_feats = np.array(ques_feats)
            ques_feats = pad_sequences(ques_feats, max_len)
            Img_feats = np.array(Img_feats)
            Img_feats = Img_feats.reshape(len(Img_feats), 4096)
            #pdb.set_trace()
            loss, acc = model.train_on_batch([Img_feats, ques_feats], labels)
            print("training Loss for epoch %d is %f with acc. %f"%(epoch, loss, acc))
            Img_feats = []
            ques_feats = []
            labels = []
            #pdb.set_trace()
            batch_ind += 1
        
        val_loss = 0
        val_acc = 0
        Img_feats = []
        ques_feats = []
        labels = []
        for img in img_val:
            img_path = 'COCO_' + 'val2014' + '_' + str(img).zfill(12) + '.jpg'
            img_feat = img_feats(IMG_DIR +'val2014/' +  img_path)
            Img_feats.append(img_feat)

        for ques in ques_val:
            ques_feat = encode_text(ques)
            ques_feats.append(ques_feat)

        for ans in ans_val:
            ans_feat = encode_ans(ans)
            labels.append(ans_feat)

        ques_feats = np.array(ques_feats)
        ques_feats = pad_sequences(ques_feats, max_len)
        Img_feats = np.array(Img_feats)
        Img_feats = Img_feats.reshape(len(Img_feats), 4096)
        loss, acc = model.test_on_batch([Img_feats, ques_feats], labels)
        print("VALIDATIOn Loss for epoch %d is %f with acc. %f"%(epoch, loss, acc))


def main():
    vqa_mlp()

if __name__ == "__main__":
    main()
