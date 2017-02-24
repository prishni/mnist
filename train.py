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

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def feedforward( activation,weights,biases):
        #print activation.shape
        zs=[]
        As=[]
        for i in range(0,2):
            w=weights[i]
            b = biases[i]
            #print activation.shape
            #print w.shape
            z=np.dot(w,activation)+b
            #bigfloat.exp(5000,bigfloat.precision(100))
            activation = sigmoid(z)
            As.append(activation)
            zs.append(z)
        #print(np.argmax(As[-1]))
        return As[-1]
def train( training_data, test_data):
    '''
    Complete this function.
    '''
    net = mytrainer.neural_net([784, 30, 10])
    biases,weights = net.create_mini_batches(training_data,test_data=test_data)
    np.save('weight1.npy', weights[0]) 
    np.save('weight2.npy', weights[1]) 
    biases[0].reshape(30,1)
    biases[1].reshape(10,1)
    np.save('bias1.npy', biases[0]) 
    np.save('bias2.npy', biases[1]) 
    

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
    w1 =  np.load('weight1.npy')
    w2 =  np.load('weight2.npy')
    b1 =  np.load('bias1.npy')
    b2 =  np.load('bias2.npy')

    weights = [w1,w2]
    biases =[b1,b2]
    results =[]
    count =0
    for (x,y) in testX:
        result_a = feedforward(x,weights,biases)
        result = np.argmax(result_a)
        if(result == y):
            count =count+1
        results.append(result)
    print (count*1.0)/100
    return results
    
