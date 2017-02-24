'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:
Roll No.:


======================================

Problem Statement:
Implement a simple 1 hidden layer MLP WITHOUT using any deep learning library
for predicting MNIST images. You are allowed to use linear algebra
libraries like numpy.

Resources:
1. https://ift6266h16.wordpress.com/2016/01/11/first-assignment-mlp-on-mnist/
2. https://github.com/tfjgeorge/ift6266/blob/master/notebooks/MLP.ipynb
    (In french. But the same repository has other useful ipython notebooks)

You might want to first code in an ipython notebook and later copy-paste
the code here.



======================================

Instructions:
1. Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/
    (four files).
2. Extract all the files into a folder named `data' just outside
    the folder containing the main.py file. This code reads the
    data files from the folder '../data'.
3. Complete the functions in the train.py file. You might also
    create other functions for your convenience, but do not change anything
    in the main.py file or the function signatures of the train and test
    functions in the train.py file.
4. The train function must train the neural network given the training
    examples and save the in a folder named `weights' in the same
    folder as main.py
5. The test function must read the saved weights and given the test
    examples it must return the predicted labels.
6. Submit your project folder with the weights. Note: Don't include the
    data folder, which is anyway outside your project folder.

Submission Instructions:
1. Fill your name and roll no in the space provided above.
2. Name your folder in format <Roll No>_<First Name>.
    For example 12CS10001_Rohan
3. Submit a zipped format of the file (.zip only).
'''

import numpy as np
import os
import train
#import bigfloat
#_stmt ::=  "" identifier ("," identifier)*
biases =[]
weights=[]
def cost_derivative(output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        yv=np.zeros(output_activations.shape)
        yv[y] = 1
        print("y = ")
        print(yv)
        return (output_activations-yv)
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class neural_net(object):
    def __init__(self, sizes):
        
        global biases
        global weights
        self.size = sizes
        for i in sizes[1:]:
            bias = np.random.randn(i,1)
            biases.append(bias)
        for i,j in zip(sizes[1:3],sizes[:2]):
            weight = np.random.randn(i,j)
            weights.append(weight)
    def feedforward(self,activation):
        global biases
        global weights
        zs=[]
        As=[]
        activation = activation.reshape(28,28)
        activation  = activation.reshape(784,1)
        As.append(activation)
        for i in range(0,2):
            w=weights[i]
            b =  biases[i]
            z=np.dot(w,activation)+b
            #bigfloat.exp(5000,bigfloat.precision(100))
            activation = 1/(1+np.exp(-1*z))
            As.append(activation)
            zs.append(z)
        #print(np.argmax(As[-1]))
        return zs,As
        #print(As[-1],max(As[-1]),np.argmax(As[-1])) 
    def feedforwardtest(self,activation):
        global biases
        global weights
        zs=[]
        As=[]
        activation = activation.reshape(28,28)
        activation  = activation.reshape(784,1)

        for i in range(0,2):
            w=weights[i]
            b =  biases[i]
            z=np.dot(w,activation)+b
            #bigfloat.exp(5000,bigfloat.precision(100))
            activation = 1/(1+np.exp(-1*z))
            As.append(activation)
            zs.append(z)
        #print(np.argmax(As[-1]))
        return As[-1]
    def backprop(self,x,y):
        global biases
        global weights
        zs,As = self.feedforward(x)
        delta = cost_derivative(As[-1],y) * sigmoid_prime(zs[-1])
        update_bias = [np.zeros(b.shape) for b in  biases]
        update_weight = [np.zeros(w.shape) for w in weights]
        update_bias[-1] =delta
        update_weight[-1]  = np.dot(delta, As[-2].transpose())
        for layer in range(2,3):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(weights[-layer+1].transpose(), delta) * sp
            update_bias[-layer] = delta
            update_weight[-layer] = np.dot(delta, As[-layer-1].transpose())
            #print("**********************")
            #print(update_bias)
        #print(update_weight[0].shape,update_weight[1].shape)
        return (update_bias, update_weight)
    
    def create_mini_batches(self,training_data_x,training_data_y):
        eta = 3.0
        mini_batch_size = 2
        epochs = 1
        mini_batches_x =[]
        mini_batches_y =[]
        n = len(training_data_x)
        for j in range(epochs):
            for k in range(0,n,mini_batch_size):
                mini_batch_x = training_data_x[k:k+mini_batch_size]
                mini_batch_y = training_data_y[k:k+mini_batch_size]
                mini_batches_x.append(mini_batch_x)
                mini_batches_y.append(mini_batch_y)
            
            for mini_batch_x,mini_batch_y in zip(mini_batches_x,mini_batches_y):
                #print("next mini batch")
                self.update_mini_batch(mini_batch_x,mini_batch_y, eta)
   
    def update_mini_batch(self,mini_batch_x,mini_batch_y,eta):
        global biases
        global weights
        #newbias =[]
        #newweight=[]
        #print(biases[0].shape)
        #print(biases[1].shape)
        newbias = [np.zeros(b.shape) for b in  biases]
        newweight = [np.zeros(w.shape) for w in weights]
        '''for b,w in ( biases,weights):
            tempb = np.zeros(b.shape)
            tempw = np.zeros(w.shape)
            print("###")
            print(tempb.shape)
            newbias.append(tempb)
            newweight.append(tempw)'''

        for x,y in zip(mini_batch_x,mini_batch_y):
            delta_b ,delta_w = self.backprop(x,y)
            #print("#####")
            #print(delta_w[0].shape,delta_w[1].shape)
            #print(newweight[0].shape,newweight[1].shape)
            #print(delta_b[0].shape,delta_b[1].shape)
            i=0
            for del_b,b in zip(newbias,delta_b):
                newbias[i] = del_b + b
                i = i+1
            i=0
            #print(delta_w[0].shape)
            for del_w,w in zip(newweight,delta_w):
                newweight[i] = del_w + w
                i=i+1
        new_final_weights=[]
        for each_w, del_w in zip(weights,newweight):
            weight1 = each_w - (eta/ len(mini_batch_x))*del_w
            #print(weight1.shape)
            new_final_weights.append(weight1)
        weights = new_final_weights
        print(new_final_weights[0])
        print("\n\n")
        print(weights[0])
        print("\n\n\n")


        new_final_biases=[]
        for each_b, del_b in  zip(biases,newbias):
            bias1 = each_b - (eta/ len(mini_batch_x))*del_b
            new_final_biases.append(bias1)
        biases = new_final_biases

    def testFunction(self,testX,testY):
        test_results = [(np.argmax(self.feedforwardtest(x)), y)
                        for (x, y) in zip(testX,testY)]
        return sum(int(x == y) for (x, y) in test_results)


def load_mnist():
    data_dir = '../data'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY


def print_digit(digit_pixels, label='?'):
    for i in range(28):
        for j in range(28):
            if digit_pixels[i, j] > 128:
                print('#', end = '')
            else:
                print ('.', end = '')
        print ('\n')

    print ('Label: ', label)




trainX, trainY, testX, testY = load_mnist()
#print "Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape
#print "\nDigit sample"
#print_digit(trainX[1], trainY[1])
net = neural_net([784, 10, 10])
'''for x,y in zip(trainX[1;200],trainY[1:200]):
    #print(x,y)
    x = x.reshape(28,28)
    x  = x.reshape(784,1)
    print(y)'''
    #print weights[2]
#print ( biases[1])

print(weights[0].shape,weights[0])
net.create_mini_batches(trainX[1:1000],trainY[1:1000])
print(weights[0].shape,weights[0])
#print weights[2]
#fp = open("biases.txt",'w')
#fp.write(biases)
#f = open("weights.txt",'w')
#f.write(weights)

print(net.testFunction(testX[1:500],testY[1:500]))
#train.train(trainX, trainY)
#labels = train.test(testX)
#accuracy = np.mean((labels == testY)) * 100.0
#print "\nTest accuracy: %lf%%" % accuracy



'''if __name__ == '__main__':
    main()'''
