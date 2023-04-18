import numpy as np


def get_dim(X):

    '''
    returns the dimensions for the given numpy array X

    input: np array X
    output: 1st and 2nd dimensions
    '''

    return X.shape[0], X.shape[1]


def init_params(xn_1, xn):

    '''
    initializes parameters W and b

    input: Dimensions/No. of Neurons of previous(xn_1) and current(xn) Layer
    output: Random weights W and b for the current layer
    '''

    W = np.random.randn(xn, xn_1)
    b = np.random.randn(xn, 1)

    return W, b


def get_max(a, b = 0):
    return a if a > b else b


def ReLU(X):
    return np.maximum(0, X)


def sigmoid(X):
    return 1/(1 + np.exp(-X))

def forward_prop(train_X_reshaped, W1, b1, W2, b2):

    '''
    Forward propagation step using the ReLU function as the 1st Activation function and 
    Sigmoid as the second activation Function.
    This is divided into 2 parts, ie for the First and the Second hidden Layer

    input: Weights, Biases of both layers
    output: Predicted Output(y_hat)
    '''

    Z1 = np.dot(W1, train_X_reshaped.T) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1.T) + b2
    A2 = sigmoid(Z2)



