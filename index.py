#Import section 
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
import regression as reg 


#Loading data from keras
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()


#Reshaping the data from 2D into 1D
train_X_reshaped = np.reshape(train_X, (train_X.shape[0], -1))
train_Y_reshaped = np.reshape(train_Y, (train_Y.shape[0], 1))


#Getting the dimensions of reshaped X
m, x0 = reg.get_dim(train_X_reshaped)


# I want 2 hidden layer of with 10 neurons each
# Hence i will initialize Weights and Biases for each layer
x1, x2 = 10, 10
W1, b1 = reg.init_params(x0, x1)
W2, b2 = reg.init_params(x1, x2)



