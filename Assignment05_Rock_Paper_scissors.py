#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:52:07 2022

@author: Nilakshi Pokharkar
"""

from keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model('keras_model.h5', compile=False)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224, 224)

#Webcam input for the model
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    
    #Get the center of the camaera image in a square shape
    h,w, _ = img.shape #640*480
    cx = int(w/2)
    hh = int(h/2)
    img = img[:,cx-hh:cx+hh]
    #print('img info: h={}, w={}'.format(h,w))
    print(img.shape)
    cv2.imshow('frame', img)
    
    #resize the iimageto 224 * 244 
    img_inf = cv2.resize(img,size)
    
    #convert cv image from BGR to RGB
    img_inf = cv2.cvtColor(img_inf, cv2.COLOR_BGR2RGB)
    
    #Normalize
    # Normalize the image: To convert between -1 to 1 range
    normalized_image_array = (img_inf.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    
    #defined the classes
    classes = ['rock', 'paper','scissors']
    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    print(classes[index])
    print(prediction)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()