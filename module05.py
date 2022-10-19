#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 23:59:08 2022

@author: Nilakshi P0kharkar
"""
#Assignment05

from MNIST_module05 import Mnist
#from PIL import Image, ImageOps #Used when code from line 22 - 24 is executed
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np

if len(sys.argv) == 3:
    file_name = sys.argv[1]
    digit_Value = sys.argv[2]
       
    file_path = 'TestImages/' + file_name
    #Open the image and reshape the image to 28 * 28 size
    # image = Image.open('2_6.png').convert('RGB')
    # image = ImageOps.fit(image, (28,28), Image.ANTIALIAS)
    # plt.imshow(image, cmap ='gray')
    # image.shape() #We will have to convert the image into numpy array when using PIL
    
    image = cv2.imread(file_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image) #Invert the colors
    image = cv2.resize(image,(28,28))
    plt.imshow(image, cmap ='gray') 
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

#Notes:
#1. While executing this file via terminal I got the error: 'QObject::moveToThread: Current thread (0x55f0f53dac50) is not the object's thread (0x55f0f53dea90). Cannot move to target thread (0x55f0f53dac50)'
#   Thus to resolve that I downgraded my opencv version from 4.6 to 4.3.0.36. Also check PyQt version it must be 5.15
#   ref: https://stackoverflow.com/questions/46449850/how-to-fix-the-error-qobjectmovetothread-in-opencv-in-python
#   ref: https://github.com/Yuliang-Liu/Curve-Text-Detector/issues/11

#2. The sys.argv is an array whose 1st parameter is always the filename. e.g: 'module05.py', '3_1.png', '3']