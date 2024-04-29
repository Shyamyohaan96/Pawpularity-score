import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
import time

DATA_DIR = "E:\machine learning\petfinder-pawpularity-score"
DATASET = "train"
DATASET_2 = "test"
IMAGESIZE = 150
dense_layers = [1]
layer_size = [64]
con_layers = [2]

data = np.loadtxt('train.csv', delimiter=',')
label = data[:, -1]



new_label = []
for j in label:
    if j > 50.0:
        new_label.append(1)
    else:
        new_label.append(0)

path = os.path.join(DATA_DIR, DATASET)
path_2 = os.path.join(DATA_DIR, DATASET_2)

i = 0
training_data = []
testing_data = []
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    new_array = cv2.resize(img_array, (IMAGESIZE, IMAGESIZE))
    paw = label[i]
    training_data.append([new_array, paw])
    i = i + 1

random.shuffle(training_data)

X = []
Y = []

for features, labels in training_data:
    X.append(features)
    Y.append(labels)

X = np.array(X).reshape(-1, IMAGESIZE, IMAGESIZE, 3)
Y = np.array(Y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

X = X/255.0

for dense_layer in dense_layers:
    for layer_siz in layer_size:
        for con_layer in con_layers:
            Model_name = "Pawpularity- {} conv- {} nodes- {} layers {}".format(con_layer, layer_siz, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs_linearreg/{}'.format(Model_name))
            print(Model_name)
            model = Sequential()

            model.add(Conv2D(layer_siz, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))

            for co in range(con_layer-1):
                model.add(Conv2D(layer_siz, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPool2D(pool_size=(2, 2)))

            model.add(Flatten())

            for de in range(dense_layer):
                model.add(Dense(layer_siz))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("linear"))

            model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

            model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.3, callbacks=[tensorboard])



