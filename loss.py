'''
Wyatt Cupp
wyattcupp@gmail.com
wyattcupp@u.boisestate.edu

This file contains a subset of common loss functions  used in artificial
neural networks.
'''

import numpy as np

class Loss:
    '''
    Base class for a Loss object.
    '''

    def loss(self, y_hat, y):
        raise NotImplementedError()

    def gradient(self, y_hat, y):
        raise NotImplementedError()

class MSE(Loss):
    '''
    Mean Squared Error loss function.
    See <https://en.wikipedia.org/wiki/Mean_squared_error>
    '''
    def loss(self, y_hat, y):
        return 0.5 * np.sum((y-y_hat)**2)

    def gradient(self, y_hat, y):
        return -(y-y_hat)

class CrossEntropy(Loss):
    '''
    Represents a Cross Entropy loss function.
    See <https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e>
    '''
    def loss(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-12, 1-1e-12)

        return -y * np.log(y_hat) - (1-y) * np.log(1-y_hat)

    def gradient(self, y_hat, y):

        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
        return - (y / y_hat) + (1 - y) / (1 - y_hat)
