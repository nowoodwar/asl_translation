'''
ASL to English Translator:
This code generates a list of predicted characters and their % confidence value based on the weights and model structure taken from the required files.
Users will sign in front of the webcam and the top three most likely predictions will be displayed over the feed
To stop the program press q

Code written in Python3 by Nate Woodward and Patrick Fuller 
5/1/2020

Requirements to run: 
	- asl_weights.h5 and asl.json files in local directory
	- all imported python packages listed below
	- a working webcam

For best results, please have only the signing hand in frame. Also, a bright, solid colored backdrop helps. 

'''

from __future__ import print_function
from common.mva19 import Estimator, preprocess
import cv2
import time
import keras
import skimage
import os
import numpy as np
from itertools import repeat
from skimage.transform import resize
from keras.models import model_from_json, load_model


character_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','space','empty']


#---------------------------- Load JSON & HDF5 file generated from Neural Network and configure the model--------------#

with open('asl.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("asl_weights.h5")
print("Loaded model from disk")
 
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #model config -- note 'accuracy' as metric of performance

model_file = "./models/mobnet4f_cmu_adadelta_t1_model.pb"
input_layer = "input_1"
output_layer = "k2tfout_0"

stride = 4
boxsize = 224

estimator = Estimator(model_file, input_layer, output_layer)

#----------------------------- Initialize OpenCV variables for webcam capture-----------------------------------------#

vidcap = cv2.VideoCapture(0)
vidcap.set(cv2.CAP_PROP_BRIGHTNESS, 0.4)
count = 0
success = True

font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (400, 400)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
charList = []


result = False

paused = True
delay = {False: 1, True: 0}


k = 0
while k != ord('q'): #keeps display of webcam feed smooth
    text = ''
    count = 0

    success,image = vidcap.read()
    #check if webcam is running
    if not success:
            raise Exception("VideoCapture.read() returned False") 

    cv2.imwrite('frame.jpg',image) #take screenshot from webcam
    img = cv2.imread('frame.jpg') # load screenshot as variable img

#----------------------------- Make predictions and return three most likely character into array for text output-----------------------#
    if img is not None:
        img = skimage.transform.resize(img, (64, 64, 3)) #resize image to fit model parameters
        input_data = np.asarray(img).reshape(-1, 64, 64, 3) #flip image
        pred = loaded_model.predict(input_data)
        pred = np.array(pred[0])
        indeces = pred.argsort()[-3:][::-1] #return three letter predictions with highest confidence  
        text = ['', '', '']
        for i in range(len(indeces)):
            text[i] = character_list[indeces[i]] + ':' + ' '*((5 + i*2) - len(character_list[indeces[i]])) + str((pred[indeces[i]]*100)//1) + '%' #format predictions to percent value
#------------------------------------------------Joint Demo Code ---------------------------------------------#

    crop_res = cv2.resize(image, (boxsize, boxsize))
    newImg, pad = preprocess(crop_res, boxsize, stride)

    tic = time.time()
    hm = estimator.predict(newImg)
    dt = time.time() - tic
    print("TTP %.5f, FPS %f" % (dt, 1.0/dt), "HM.shape ", hm.shape)

    hm = cv2.resize(hm, (0, 0), fx=stride, fy=stride)
        #bg = cv2.normalize(hm[:, :, -1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    viz = cv2.normalize(np.sum(hm[:, :, :-1], axis=2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        #cv2.imshow("Background", bg)
        
    #   cv2.imshow("Input frame", frame)

 


#------------------------------ Draw shape to assist in readabilty and overlay top three predictions with confidence % ---------------#
    cv2.rectangle(image, (440,300), (700,500), (0,0,0), -100)

    if count % 10 == 0: #update predictions every 10 iterations
        cv2.putText(image, text[0], (450, 350), font, .9*fontScale, fontColor, lineType)
        cv2.putText(image, text[1], (450, 400), font, .8*fontScale, fontColor, lineType)
        cv2.putText(image, text[2], (450, 450), font, .7*fontScale, fontColor, lineType)
    cv2.imshow("All joint heatmaps", viz)
    cv2.imshow('Start Signing',image) #webcam feed display
    k = cv2.waitKey(delay[paused])

    if k & 0xFF == ord('p'):
        paused = not paused

    count += 1
#---------------------------------------------- Garbage clean up--------------------------------------------------#
vidcap.release()
cv2.destroyAllWindows()
