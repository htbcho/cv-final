import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import applications
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from flask import Flask, request, make_response, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
import os
import urllib.request
from PIL import Image
import requests
from io import StringIO
import cv2
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import time
import uuid
import glob



app = Flask(__name__) #, static_folder='.'
app._static_folder = os.path.abspath("templates/static/")
CORS(app)


test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

prediction = ''

@app.route('/')
def index():
    return render_template('/index.html', title='Learn ASL!')

img_dir = 'webcam_images/curr.jpg'
global start
start = 0

global url
url = ''

global count
count = 0

@app.route('/model', methods = ['POST', 'GET'])
def predict():
    ret = ''
    if request.method == 'POST':
        global url
        url = request.form['url']
        global start
        start = 1
    # global start
    if start == 1:
        global count
        count+=1
        time.sleep(1)
        i = str(uuid.uuid4())
        imgdir = 'webcam_images/curr' + i + '.jpg'
        urllib.request.urlretrieve(url, imgdir)
        img = image.load_img(imgdir, target_size=(64,64))
        # print(os.)
        img_array = image.img_to_array(img)
        # img_array = np.divide(img_array, 255.0)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        loaded_model = tf.contrib.saved_model.load_keras_model('./scratch_tmp_dir/1556848947/') # THE WORKING MODEL
        # loaded_model = load_model('vgg_model.h5') # VGG16
        # loaded_model = tf.contrib.saved_model.load_keras_model('./tmp_dir/1556737843/') # MOBILENET
        # preprocessed_image = applications.mobilenet.preprocess_input(img_array_expanded_dims) # MOBILENET

        result = loaded_model.predict(img_array_expanded_dims)
        prediction = labels[np.argmax(result)]
        ret = { 'label': prediction }
        if count > 20:
            filelist = [ f for f in os.listdir('webcam_images') if f.endswith(".jpg") ]
            for f in filelist:
                os.remove(os.path.join('webcam_images/', f))
            count = 0
        # os.remove(img_dir)
        return jsonify(ret)



def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response
app.after_request(add_cors_headers)
# r = requests.post('/index.js', data=prediction)


if __name__ == '__main__':
    app.run(debug=True) #host='127.0.0.1', port=8000
