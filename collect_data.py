import cv2     # for capturing videos
import math   # for mathematical operations
#import pandas as pd
#from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
#from keras.utils import np_utils
from skimage.transform import resize   # for resizing images


count = 0
videoFile = "/Users/ella/Desktop/test-Z.mov"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) #frame rate
#print(frameRate)
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    #print(frameId)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % 5 == 0):
        filename ="test/Z/test_frame%d.jpg" % count;count+=1
        frame = frame[:,280:1000, :]
        frame = resize(frame, preserve_range=True, output_shape=(200,200)).astype(int)
        cv2.imwrite(filename, frame)
cap.release()

print ("Done!")
