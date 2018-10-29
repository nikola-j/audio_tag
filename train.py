import pickle

import numpy as np
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.utils import shuffle


# class DataProc(Sequence):
#     def __init__(self, x, y, bs, cutoff):
#         self.x = x
#         self.y = y
#         self.bs = bs
#         self.cutoff = cutoff
#
#     def __getitem__(self, index):
#         start = index * self.bs
#         end = (index + 1) * self.bs
#         x_res = []
#         for i in range(start, end):
#             x_res.append(self.x[i])
#
#         x_res = np.stack(x_res, axis=0)
#         y_res = np.array(self.y[start:end])
#
#         return x_res, y_res
#
#     def __len__(self):
#         return len(self.x // self.bs)


def define_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=128))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', input_dim=128))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(41, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=[
                  'accuracy', 'top_k_categorical_accuracy'])
    return model


def train_model():
    # Import dataset
    with open("audioset/output.p", 'rb') as infile:
        dataset = pickle.load(infile)

    x, y = dataset
    for i in range(len(x)):
        x[i] = np.mean(x[i], axis=0)

    x, y = shuffle(x, y)

    x = np.array(x)
    y = np.array(y)

    y = to_categorical(y)

    print(x.shape, y.shape)
    # Define model
    model = define_model()

    model.fit(x, y, batch_size=4096, epochs=10000, validation_split=0.9)


if __name__ == '__main__':
    train_model()
