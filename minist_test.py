import numpy as np
from loss import SoftmaxCrossEntropy
from utils.tools import rel_error

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import matplotlib.pyplot as plt
from applications import MNISTNet
from loss import SoftmaxCrossEntropy, L2
from optimizers import Adam
from utils.datsets import MNIST

import warnings
warnings.filterwarnings('ignore')


mnist = MNIST()
mnist.load()

model = MNISTNet()
loss = SoftmaxCrossEntropy(num_class=10)



# define your learning rate sheduler
def func(lr, iteration):
    if iteration % 1000 ==0:
        return lr*0.5
    else:
        return lr


adam = Adam(lr=0.001, decay=0,  sheduler_func = func)
l2 = L2(w=0.001) # L2 regularization with lambda=0.001

model.compile(optimizer=adam, loss=loss, regularization=l2)
train_results, val_results, test_results = model.train(
    mnist, 
    train_batch=64, val_batch=1000, test_batch=1000, 
    epochs=1, 
    val_intervals=100, test_intervals=300, print_intervals=100)


print('*************')
xtest,ytest=mnist.get_test()
yhat=model.predict_prob(xtest,ytest)

print(yhat[:5])