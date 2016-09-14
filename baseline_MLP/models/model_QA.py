
from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Merge
import keras.backend as K
import os


def MeanPool1D(x):
    y = K.mean(x, axis=1)
    return y

def MeanPool1D_shape(input_shape):
    res = list((input_shape[0], input_shape[2]))
    return tuple(res)


def modelQA(vocab_size, img_dim, wordvec_dim, inp_len, embeddings):
    # Returns the QA model
    x = Input(shape=(4096,))
    img_model1 = Model(input=x, output=x)
    img_model2 = Sequential()
    img_model2.add(img_model1)
    img_model2.add(Dense(300))


    text_model = Sequential()
    text_model.add(Embedding(vocab_size, wordvec_dim, weights=[embeddings], input_length=inp_len, trainable=False))#, mask_zero=True))
    text_model.add(Lambda(MeanPool1D, output_shape=MeanPool1D_shape))
    text_model.add(Dense(300))

    model = Sequential()
    model.add(Merge([img_model2, text_model], mode='mul'))
    model.add(Dropout(0.25))
    model.add(Dense(300))
    model.add(Dropout(0.25))
    model.add(Dense(300))
    model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', \
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = modelQA(1000, 100, 300)
    print(model)
