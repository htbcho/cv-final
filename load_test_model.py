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


app = Flask(__name__) #, static_folder='.'
app._static_folder = os.path.abspath("templates/static/")
CORS(app)


test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

prediction = ''

@app.route('/', methods=['GET'])
def index():
    return render_template('/index.html', title='Learn ASL!')

# img_dir = 'webcam_images/curr.jpg'

@app.route('/model', methods = ['POST']) #what is a file path
def predict():
    url = request.form['url']
    urllib.request.urlretrieve(url, 'webcam_images/curr.jpg')
    print("got image")
    print(os.listdir('webcam_images'))
    print(' - - - - - - - ')
    img = image.load_img('webcam_images/curr.jpg', target_size=(64,64))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    loaded_model = tf.contrib.saved_model.load_keras_model('./scratch_tmp_dir/1556848947/')
    # preprocessed_image = applications.mobilenet.preprocess_input(img_array_expanded_dims)
    result = loaded_model.predict(img_array_expanded_dims)
    prediction = labels[np.argmax(result)]

    ret = { 'label': prediction }

    # print('after model')
    # print(os.listdir('webcam_images'))
    # os.remove(img_dir)
    # print(os.listdir('webcam_images'))
    return jsonify(ret)


    # print(filename)
    # print(np.shape(result))
    # print(labels[np.argmax(result)])

    # if request.method == 'POST':
    #     result = prediction
    #     resp = make_response('{"response": '+prediction+'}')
    #     resp.headers['Content-Type'] = "application/json"
    #     return jsonify(resp)

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
