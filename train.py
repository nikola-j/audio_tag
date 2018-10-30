import argparse
import pickle

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.utils import shuffle


def define_model_large():
    inputs = Input((128,))
    x1f = Dense(128, activation='relu')(inputs)
    x1b = BatchNormalization()(x1f)
    x1d = Dropout(0.3)(x1b)
    x1 = keras.layers.add([x1d, inputs])
    x2f = Dense(128, activation='relu')(x1)
    x2b = BatchNormalization()(x2f)
    x2d = Dropout(0.3)(x2b)
    x2 = keras.layers.add([x1, x2d, inputs])
    x3f = Dense(64, activation='relu')(x2)
    x3b = BatchNormalization()(x3f)
    x3d = Dropout(0.3)(x3b)
    x4f = Dense(64, activation='relu')(x3d)
    x4b = BatchNormalization()(x4f)
    x4d = Dropout(0.3)(x4b)
    x4 = keras.layers.add([x3d, x4d])
    x5f = Dense(64, activation='relu')(x4)
    x5b = BatchNormalization()(x5f)
    x6f = Dense(41, activation='softmax')(x5b)

    model = Model(inputs=inputs, outputs=x6f)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01),
                  metrics=['accuracy', ])
    return model


def define_model_small():
    inputs = Input((128,))
    x1f = Dense(128, activation='relu')(inputs)
    x1b = BatchNormalization()(x1f)
    x1d = Dropout(0.3)(x1b)
    x2f = Dense(64, activation='relu')(x1d)
    x2b = BatchNormalization()(x2f)
    x2d = Dropout(0.3)(x2b)
    x2f = Dense(41, activation='softmax')(x2d)
    model = Model(inputs=inputs, outputs=x2f)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01),
                  metrics=['accuracy', ])
    return model


class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print("lr:", K.eval(lr), " ", end='')


def format_dataset(x, y):
    full_x, full_y = [], []

    for elem_x, elem_y in zip(x, y):
        for sec_x in elem_x:  # Create new data point for each second of wav, set the label as same
            full_x.append(sec_x)
            full_y.append(elem_y)

    full_x, full_y = shuffle(full_x, full_y)

    full_x = np.array(full_x)
    full_y = np.array(full_y)

    full_y = to_categorical(full_y)

    return full_x, full_y


def train_model():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, default='large', help='What model to train')
    arg_parser.add_argument('--batch_size', type=int, default=1024, help='What batch size to use')
    args = arg_parser.parse_args()

    # Limit gpu usage:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # Import dataset
    with open("audioset/output.p", 'rb') as infile:
        dataset = pickle.load(infile)

    x, y = dataset
    x, y = shuffle(x, y)

    split_point = int(0.9 * len(x))

    x_train, y_train = format_dataset(x[:split_point], y[:split_point])
    x_val, y_val = format_dataset(x[split_point:], y[split_point:])

    print(x_train.shape, x_val.shape)
    # Define model
    if args.model == "small":
        model = define_model_small()
    elif args.model == "large":
        model = define_model_large()
    else:
        model = None
        print("Please define model")
        exit()

    rlrop = ReduceLROnPlateau('loss', factor=0.8, patience=10, min_lr=0.000001, cooldown=10)
    prl = PrintLearningRate()
    # Fit data
    model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              batch_size=args.batch_size,
              epochs=1000,
              validation_split=0.9, verbose=2, callbacks=[rlrop, prl])

    # Save model
    model.save("trained_model_small.ckpt")


if __name__ == '__main__':
    train_model()
