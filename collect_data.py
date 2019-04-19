import cv2     # for capturing videos
import math   # for mathematical operations
#import pandas as pd
#from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
#from keras.utils import np_utils
from skimage.transform import resize   # for resizing images


count = 0
videoFile = "Untitled.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) #frame rate
print(frameRate)
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    print(frameId)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % 5 == 0):
        filename ="data/frame%d.jpg" % count;count+=1
        print(type(frame))
        frame = resize(frame, preserve_range=True, output_shape=(224,224)).astype(int)
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")
