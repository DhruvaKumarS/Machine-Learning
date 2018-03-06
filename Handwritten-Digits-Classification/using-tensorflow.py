'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize
import sys
import time

iter = 1
# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1/(1 + np.exp(np.multiply(-1,z)))    # your code here
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    global iter
    iter = iter + 1
    
    training_data = np.insert(training_data, training_data.shape[1], 1, axis=1)
    hidden_layer_output = sigmoid(np.dot(training_data, np.transpose(w1)))
    hidden_layer_output = np.insert(hidden_layer_output, hidden_layer_output.shape[1], 1, axis=1)
    output_layer_output = sigmoid(np.dot(hidden_layer_output, np.transpose(w2)))
    
    training_label_matrix = np.zeros(shape=(training_label.shape[0], 2))
    training_label = training_label.astype(int)
    training_label_matrix[np.arange(len(training_label_matrix)), training_label] = 1
    
    error_output = -1 * np.sum(training_label_matrix * np.log(output_layer_output) + (1 - training_label_matrix) * np.log(1 - output_layer_output))
    error = np.sum(error_output)/training_label.shape[0]
    delta_l = output_layer_output - training_label_matrix
    hidden_layer_output_minus_bias = np.delete(hidden_layer_output, hidden_layer_output.shape[1] - 1, axis=1)
    derivative_w2 = np.dot(np.transpose(delta_l), hidden_layer_output_minus_bias)
    
    delta_l_w2 = np.dot(delta_l, np.delete(w2, w2.shape[1] - 1, axis=1)) * (1 - hidden_layer_output_minus_bias) * hidden_layer_output_minus_bias
    derivative_w1 = np.dot(np.transpose(delta_l_w2) , np.delete(training_data, training_data.shape[1] - 1, axis=1))
    
    obj_val = error + lambdaval * (np.sum(np.sum(w2 * w2, axis=0)) + np.sum(np.sum(w1 * w1, axis=0))) / (2 * training_data.shape[0])
        
    derivative_w2 = np.insert(derivative_w2, derivative_w2.shape[1], 1, axis=1)
    derivative_w1 = np.insert(derivative_w1, derivative_w1.shape[1], 1, axis=1)
    grad_w2 = (derivative_w2 + lambdaval * w2) / training_data.shape[0]
    grad_w1 = (derivative_w1 + lambdaval * w1) / training_data.shape[0]

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    data = np.insert(data, data.shape[1], 1, axis=1)
    hidden_layer_output = sigmoid(np.dot(data, np.transpose(w1)))
    hidden_layer_output = np.insert(hidden_layer_output, hidden_layer_output.shape[1], 1, axis=1)
    output_layer_output = sigmoid(np.dot(hidden_layer_output, np.transpose(w2)))
    labels = np.argmax(output_layer_output, axis=1)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('./face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255
    
    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 15;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

start_time = time.time()

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)


#find the accuracy on Training Dataset
print('Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
print('Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
print('Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
