'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:Prishni Rateria
Roll No.:16CS60R58

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import mytrainer

def train(trainX, trainY):
    '''
    Complete this function.
    '''
    training_data, test_data = mytrainer.load_data_format()
    net = mytrainer.neural_net([784, 30, 10])
    net.create_mini_batches(training_data, 30, 10, 3.0, test_data=test_data)

    


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    accuracy,result = mytrainer.testFunction(test_data)
    return result
