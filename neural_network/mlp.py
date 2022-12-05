'''
Wyatt Cupp
wyattcupp@gmail.com
'''
import math
import numpy as np
import sklearn.metrics as metrics

import activations
import util


class MultiLayerPerceptron:
    '''
    Represents an MLP model.
    '''

    def __init__(self, layers, loss, batch_size=32, verbose=False):
        self.layers = self._verify_layers(layers)
        self.loss = loss
        self.batch_size = batch_size
        self.verbose = verbose

        self._init_params()

    def _verify_layers(self, layers):
        if len(layers) < 2:
            raise ValueError("MLP must have at least 2 layers.")

        if layers[0].input_shape is None:
            raise ValueError("Input shape required for first input layer.")

        return layers

    def _init_params(self):
        '''
        Set initial weights and bias to all layers.
        '''
        # verify first input layer:
        if self.layers[0].input_shape is None:
            raise ValueError(
                'First Layer object must contain input_shape parameter.')

        # set input weights and bias:
        self.layers[0].init_parameters()

        for i in range(1, len(self.layers)):
            # set input shape of curr layer to output shape of prev layer (dot produt rules)
            self.layers[i].input_shape = self.layers[i-1].units
            self.layers[i].init_parameters()

    def _check_input_dims(self, input_shape, X):
        '''
        Cross checks input dimensions from first layer and given X dimensions.
        If the given X dimensions don't match, reshape occurs.
        '''
        if input_shape != X.shape[0] and input_shape != X.shape[1]:
            raise ValueError(
                'X does not have the appropriate shape: {}'.format(X.shape))

        if input_shape != X.shape[1]:
            # flip input shape to match our networks architecture:
            X = X.T

        return X

    def train(self, X, y, lr=0.001, epochs=300):
        '''
        Trains the model with given parameters.
        '''
        X = self._check_input_dims(self.layers[0].input_shape, X)

        if self.verbose:
            print('Beginning training ...')
            for i in range(len(self.layers)):
                print('Layer {} weights shape: {}'.format(
                    i, self.layers[i].W.shape))
            print('Learning Rate: {}, batch size: {}, epochs: {}'.format(
                lr, self.batch_size, epochs))
            print('---------------------------------------------------\n')

        cost_iter = []
        for i in range(epochs):
            # iterate over batches and train:
            batch_cost = []
            for X_batch, y_batch in util.batch_iterator(X, y, batch_size=self.batch_size):
                # forward propagation: run through a forward pass for given batch
                y_hat = self._forward_propagation(X_batch)

                # calculate the cost for current batch in current epoch iteration
                cost = np.mean(self.loss.loss(y_hat, y_batch))
                batch_cost.append(cost)

                # perform back propagation and update weights
                self._gradient_descent(X_batch, y_batch, y_hat, lr)

            # append total mean cost of batches
            cost_iter.append(np.mean(batch_cost))

            if self.verbose and i % 100 == 0:
                print('Cost after epoch #{}: {}'.format(i, cost))
        return cost_iter

    def _forward_propagation(self, X):
        '''
        Propagate through layers and return pred value y_hat.
        '''
        y_hat = X
        for layer in self.layers:
            y_hat = layer.forward(y_hat)

        return y_hat

    def _gradient_descent(self, X, y, y_hat, lr=0.001):
        '''
        Performs gradient descent.
        See <https://en.wikipedia.org/wiki/Gradient_descent>
        '''
        # Output error gradient
        E = self.loss.gradient(y_hat, y)
        # common gradient to all subsequent layer calcs (used for memoization)
        chained_grads = E

        # loop through layers starting from last, backpropogate, update weights/bias
        for layer in reversed(self.layers):
            chained_grads = layer.backward(chained_grads, lr)

    def predict(self, X):
        '''
        Generates label predictions based on input X.
        Params:
        X: np.array
            n-dimensional array of inputs.
        '''
        # TODO: check X dims prior to predicting
        y_hat = np.squeeze(self._forward_propagation(X))

        if self.layers[len(self.layers)-1].activation is None:
            if self.layers[len(self.layers)-1].units == 1:
                return activations.Sigmoid().predict(y_hat)  # binary classification
            else:
                return activations.SoftMax().predict(y_hat)  # multi classification

        return self.layers[len(self.layers)-1].activation.predict(y_hat)

    def accuracy(self, X, y):
        '''
        Returns the accuracy score for the given input and labels.
        '''
        preds = self.predict(X)

        return metrics.accuracy_score(y, preds)

    def info(self, name=None):
        '''
        Returns a summary about the network.
        '''
        # TODO
        pass


class DenseLayer:
    '''A fully-connected MLP Neural Network Layer object.
    Params:
    units: int
        Number of nodes in this layer. Can also be considered output shape.
    activation: ActivationFunction
        The activation function to be used on this layer. See activations.py.
    input_shape:
        Input shape for first layer. Must be provided to constructor for first 
        input layer of the network.
    '''

    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.W = None  # weights
        self.B = None  # bias
        self.X = None  # layer input
        self.Z = None  # layer Z matrix
        self.A = None  # layer output y_hat

    def init_parameters(self):
        '''
        Initialize weights and bias.
        '''
        limit = 1 / math.sqrt(self.input_shape)  # helps avoid overflow
        self.W = np.random.uniform(-limit, limit,
                                   size=(self.input_shape, self.units))
        self.B = np.zeros(shape=(1, self.units))

    def forward(self, X):
        '''
        Represents a forward pass for the current layer (y=XW+b).
        Params:
        X:
            Input data to the current forward pass of this Layer object.
        '''
        self.X = X
        self.Z = np.dot(X, self.W) + self.B
        self.A = self.activation.activate(self.Z)

        return self.A

    def backward(self, chained_grads, lr):
        '''
        Represents a backward pass for current layer.
        Params:
        chained_grads: 
            Gradients previously calculated (backwards) via the chain rule. 
            These gradients provide memoization to save unravelling the chain
            rule and recomputation every single time this method is called.
        lr:
            Learning rate applied to weights and bias updates (default=0.001 from train())
        '''
        chained_grads = np.multiply(
            chained_grads, self.activation.activate(self.Z, gradient=True))

        W = self.W  # reference for chained_grads return (before update)

        dW = np.dot(self.X.T, chained_grads)
        dB = np.sum(chained_grads, axis=0, keepdims=True)

        # update this layers weights/bias
        self.W = self.W - (lr*dW)
        self.B = self.B - (lr*dB)

        # Return accumulated gradient for next layer
        return np.dot(chained_grads, W.T)

    def print_layer(self):
        '''
        Prints the attributes of the layer.
        '''
        print("W: {}\nW_shape: {}\n".format(self.W, self.W.shape))
        print("Z: {}\nZ_shape: {}\n".format(self.Z, self.Z.shape))
        print("A: {}\nA_shape: {}\n".format(self.A, self.A.shape))
        print("B: {}\nB_shape: {}\n".format(self.B, self.B.shape))
