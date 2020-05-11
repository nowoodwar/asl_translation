from common.mva19 import Estimator, preprocess
import cv2
import keras
import os
import numpy as np

model_file = "./models/mobnet4f_cmu_adadelta_t1_model.pb"
input_layer = "input_1"
output_layer = "k2tfout_0"
folder = 'asl_alphabet_train/'

stride = 4
boxsize = 224

estimator = Estimator(model_file, input_layer, output_layer)

for folderName in os.listdir(folder):
    if not folderName.startswith('.'):
        for imageName in os.listdir(folder + folderName):
            img = cv2.imread(folder+folderName+'/'+imageName)
            crop_res = cv2.resize(img, (boxsize, boxsize))
            newImg, _ = preprocess(crop_res, boxsize, stride)
            
            heatmap = estimator.predict(newImg)
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride)
            bg = cv2.normalize(heatmap[:, :, -1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imwrite(folder+folderName+'/'+imageName, bg)