'''
Wyatt Cupp
wyattcupp@gmail.com
wyattcupp@u.boisestate.edu

This file defines various activation functions to be used in a neural netowrk.
'''
import numpy as np

class ActivationFunction:
    '''
    Base class for activation functions. This class houses the activate method, 
    as well as the activation specific accuracy algorithm needed.

    See <https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6>
    '''
    def activate(self, Z, gradient=False):
        raise NotImplementedError()

    def predict(self, y_hat):
        '''
        To be implemented only by functions used in an output layer.
        '''
        raise NotImplementedError()

class Sigmoid(ActivationFunction):
    '''
    Represents the sigmoid activation function.
    '''

    def activate(self, Z, gradient=False):
        '''
        Computes and returns the sigmoid of the input x.
        '''

        if gradient:
            Z = self.activate(Z, gradient=False)
            return Z * (1 - Z)
        return 1/(1 + np.exp(-Z))
    
    def predict(self, y_hat):
        '''
        Generates binary classification preds for given y_hat (sigmoid output)
        '''
        preds = 1. * (y_hat > 0.5)

        return preds        

class Relu(ActivationFunction):
    '''
    Represents the ReLU activation function.
    '''

    def activate(self, Z, gradient=False):
        '''
        Relu activation function.
        '''
        if gradient:
            return np.where(Z>=0, 1, 0)
    
        return np.where(Z>=0,Z,0)

class LeakyRelu(ActivationFunction):
    '''
    Represents the Leaky ReLU activation function.

    See <https://keras.io/api/layers/activation_layers/leaky_relu/>
    '''
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def activate(self, Z, gradient=False):
        '''
        Leaky ReLU activation function.
        '''
        if gradient:
            return np.where(Z>=0, 1, self.alpha)
    
        return np.where(Z>=0,Z, self.alpha*Z)

class TanH(ActivationFunction):
    '''
    Represents the TanH activation function.
    '''

    def activate(self, Z, gradient=False):
        '''
        Tanh activation function.
        '''
        if gradient:
            Z = self.activate(Z, gradient=False)
            return 1-np.power(Z,2)

        return 2 / (1+np.exp(-2*Z))-1

class SoftMax(ActivationFunction):
    '''
    Represents teh Softmax activation function.
    '''

    def activate(self, Z, gradient=False):
        '''
        Softmax activation function.
        '''
        if gradient:
            Z = self.activate(Z, gradient=False)
            return Z * (1-Z)

        e_Z = np.exp(Z-np.max(Z, axis=-1, keepdims=True))
        return e_Z / np.sum(e_Z, axis=-1, keepdims=True)
    
    def predict(self, y_hat):
        '''
        Makes a prediction by returning index of most probable feature.
        '''
        return np.argmax(y_hat, axis=1)

