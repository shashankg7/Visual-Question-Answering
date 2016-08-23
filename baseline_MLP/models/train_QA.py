
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

IMG_DIR = '/home/shashank/data/VQA/dataset/VQAorg/Images/train2014/'

def vqa_mlp(batch_size=32, epochs=10, max_len=5):
    vgg = VGG16(include_top=True, weights='imagenet')
    gen_vocab()
    embeddings = load_embeddings()
    vocab_size = get_vocab_size()
    model = modelQA(vocab_size, 4096, 100, max_len, embeddings)


    ques_train, ans_train, img_train, ques_val, ans_val, img_val = parse_QA()
    # Parse all training images and load them into memory
    for epoch in xrange(epochs):
        batch_ind = 1
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
            #img_batch = [img_train[x] for x in batch_index]
            ques_batch = ques_train[batch_ind * batch_size: (batch_ind + 1) * batch_size]
            #ques_batch = [ques_train[x] for x in batch_index]
            ans_batch = ans_train[batch_ind * batch_size: (batch_ind + 1) * batch_size]
            #ans_batch = [ans_train[x] for x in batch_index]

            for img in img_batch:
                img_path = 'COCO_' + 'train2014' + '_' + str(img).zfill(12) + '.jpg'
                img_feat = img_feats(IMG_DIR + img_path)
                Img_feats.append(img_feat)

            for ques in ques_batch:
                ques_feat = encode_text(ques)
                ques_feats.append(ques_feat)

            for ans in ans_batch:
                ans_feat = encode_ans(ans)
                labels.append(ans_feat)

            ques_feats = np.array(ques_feats)
            ques_feats = pad_sequences(ques_feats, 5)
            Img_feats = np.array(Img_feats)
            Img_feats = Img_feats.reshape(batch_size, 4096)
            # pdb.set_trace()
            loss, acc = model.train_on_batch([Img_feats, ques_feats], labels)
            print("Loss for epoch %d is %f with accuracy %f"%(epoch, loss, acc))
            Img_feats = []
            ques_feats = []
            labels = []
            #pdb.set_trace()
            batch_ind += 1


def main():
    vqa_mlp()


if __name__ == "__main__":
    main()
