#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:03:37 2022

@author: Nilakshi Pokharkar
"""
#%%
import urllib.request
import gzip
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

class Mnist():
    img_size = 784
    img_Dim = (1, 28 ,28)
    # 'sample_weight.pkl' contains weights and biases
    weight_file_Name = 'sample_weight.pkl'
     
    def __init__(self):
        self.key_file = {
                'train_img':'train-images-idx3-ubyte.gz',
                'train_label':'train-labels-idx1-ubyte.gz',
                'test_img': 't10k-images-idx3-ubyte.gz',
                'test_label': 't10k-labels-idx1-ubyte.gz'
            }
        
    # Download the datasets from the URL in the current working repository
    def download_dataset(self):
        url_base = 'http://yann.lecun.com/exdb/mnist/'
        
        for value in self.key_file.values():
            if os.path.exists(value):
                print('File exists!')
            else:
                print('Downloading {}.....'.format(value))
                urllib.request.urlretrieve(url_base + value, value)
                print('Download complete!!!')
                
    def load_images(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16) #This data can be found in the dataset URL
        images = images.reshape(-1, self.img_size)
        
        print('Done with image loading: ', file_name)
        return images        
    
    def load_labels(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        print('Done with label load: ', file_name)
        return labels
    
    def sigmoid_func(self, a):
        return 1/(1+np.exp(-a))
    
    # Refined softmax function (is good even if a is very big)
    def softmax_func(self, a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a/sum_exp_a
        return y
    
    #Loads the weights and biases from the weight_file_Name to the network variable
    def init_network(self):
        with open(self.weight_file_Name,'rb') as f:
            network = pickle.load(f)
            
        return network

    #Predict using the network
    def predict(self, network, x):
        w1, w2, w3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid_func(a1)
        
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid_func(a2)
        
        a3 = np.dot(z2, w3) + b3
        z3 = self.softmax_func(a3)
        #print(z3)
        return z3
    

if __name__ == '__main__':
    import sys
    import cv2
    
    if len(sys.argv) == 3:
        file_name = sys.argv[1]
        digit_Value = sys.argv[2]
    
        file_path = 'TestImages/' + file_name
        #Open the image and reshape the image to 28 * 28 size
        # image = Image.open('2_6.png').convert('RGB')
        # image = ImageOps.fit(image, (28,28), Image.ANTIALIAS)
        # plt.imshow(image, cmap ='gray')
        # image.shape()
    
        image = cv2.imread(file_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(image) #Invert the colors
        image = cv2.resize(image,(28,28))
        plt.imshow(image, cmap ='gray') 
        plt.plot()
        #image = np.reshape(image, (-1, 784))
        image = image.reshape(784,) #Reshaped the img in the 784 size vector!!
        #image = cv2.resize(image,(28,28))
        #cv2.imshow('grayscale',image)
    
        mnist = Mnist()
        network = mnist.init_network()
        y = mnist.predict(network, image)
        predicted_num = np.argmax(y)
        
        if(predicted_num == digit_Value):
            print('Image {} is for digit {} is recognized as {}'.format(file_name, digit_Value, predicted_num))
        else:
            print('Image {} is for digit {} but the inference result is {}'.format(file_name, digit_Value, predicted_num))