import random
import os
import numpy as np
import cPickle
import gzip

'''Loader downloaded from
    neural network and deep learning book chap 1'''
def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_format():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(y):
    yp = np.zeros((10, 1))
    yp[y] = 1.0
    return yp

class neural_net(object):

    def __init__(self, sizes):
        np.random.seed(100)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []
        for i in sizes[1:]:
            bias = np.random.randn(i,1)
            self.biases.append(bias)
        for i,j in zip(sizes[1:3],sizes[:2]):
            weight = np.random.randn(i,j)
            self.weights.append(weight)

    def feedforward(self, activation):
        
        zs=[]
        As=[]
        for i in range(0,2):
            w=self.weights[i]
            b = self.biases[i]
            z=np.dot(w,activation)+b
            #bigfloat.exp(5000,bigfloat.precision(100))
            activation = sigmoid(z)
            As.append(activation)
            zs.append(z)
        #print(np.argmax(As[-1]))
        return As[-1]

    def feedforwardtrain(self,activation):
        zs=[]
        As=[]
        As.append(activation)
        for i in range(0,2):
            w=self.weights[i]
            b =  self.biases[i]
            z=np.dot(w,activation)+b
            #bigfloat.exp(5000,bigfloat.precision(100))
            activation = sigmoid(z)
            As.append(activation)
            zs.append(z)
        #print(np.argmax(As[-1]))
        return zs,As
    def create_mini_batches(self, training_data, test_data):
    	epochs =30
    	mini_batch_size =10
    	eta =3.0
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            #random.shuffle(training_data)
            mini_batches =[]
            
            for k in xrange(0, n,mini_batch_size):
                if( k+mini_batch_size >n):
                    break
                mini_batches.append( training_data[k:k+mini_batch_size])
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("next")
                self.testFunction(test_data)
            else:
                pass
            #k = k+ mini_batch_size

    def update_mini_batch(self,mini_batch, eta):
        
        newbias =[]
        newweight=[]
        for b in  self.biases:
        	newbias.append(np.zeros(b.shape))
        
        for w in self.weights:
        	newweight.append(np.zeros(w.shape))
        	

        for x, y in mini_batch:
            #print x.shape
            delta_b ,delta_w = self.backprop(x,y)
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
        for each_w, del_w in zip(self.weights,newweight):
            weight1 = each_w - (eta/ len(mini_batch))*del_w
            #print(weight1.shape)
            new_final_weights.append(weight1)
        self.weights = new_final_weights

        new_final_biases=[]
        for each_b, del_b in  zip(self.biases,newbias):
            bias1 = each_b - (eta/ len(mini_batch))*del_b
            new_final_biases.append(bias1)
        self.biases = new_final_biases

    def backprop(self, x, y):
        
        update_bias =[]
        update_weight=[]
        for b in  self.biases:
        	update_bias.append(np.zeros(b.shape))
        
        for w in self.weights:
        	update_weight.append(np.zeros(w.shape))
        zs,As = self.feedforwardtrain(x)
        delta = (As[-1]-y) * sigmoid(zs[-1])*(1-sigmoid(zs[-1]))
        update_bias =[]
        update_weight=[]
        for b in  self.biases:
        	update_bias.append(np.zeros(b.shape))
        
        for w in self.weights:
        	update_weight.append(np.zeros(w.shape))
        
        update_bias[-1] =delta
        update_weight[-1]  = np.dot(delta, As[-2].T)
        for layer in range(2,3):
            z = zs[-layer]
            sp = sigmoid(z)*(1-sigmoid(z))
            delta = np.dot(self.weights[-layer+1].T, delta) * sp
            update_bias[-layer] = delta
            update_weight[-layer] = np.dot(delta, As[-layer-1].transpose())
            #print("**********************")
            #print(update_bias)
        #print(update_weight[0].shape,update_weight[1].shape)
        return (update_bias, update_weight)

    def testFunction(self,test_data):
    	count=0
        results =[]
    	for (x, y) in test_data:
    		result_a = self.feedforward(x)
    		result = np.argmax(result_a)
            results.append(result)
        	if(result == y):
        		count = count+1
        return count,results

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

training_data, v_data,test_data = load_data_format()
net = neural_net([784, 30, 10])
net.create_mini_batches(training_data, test_data=test_data)

