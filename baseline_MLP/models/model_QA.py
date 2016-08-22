
from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
from keras.layers.core import Merge
import keras.backend as K


def MeanPool1D(x):
    y = K.mean(x, axis=1)
    return y

def MeanPool1D_shape(input_shape):
    res = list((input_shape[0], input_shape[2]))
    return tuple(res)


def modelQA(vocab_size, img_dim, wordvec_dim, inp_len):
    # Returns the QA model
    x = Input(shape=(4096,))
    img_model = Model(input=x, output=x)

    text_model = Sequential()
    text_model.add(Embedding(vocab_size, wordvec_dim, input_length=inp_len))#, mask_zero=True))
    text_model.add(Lambda(MeanPool1D, output_shape=MeanPool1D_shape))

    model = Sequential()
    model.add(Merge([img_model, text_model], mode='concat'))
    model.add(Dense(100))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', \
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = modelQA(1000, 100, 300)
    print(model)