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
# import os
from tensorflow.keras.models import load_model
# import requests
from flask import Flask, request, make_response, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
import os
from PIL import Image
import requests
from StringIO import StringIO


app = Flask(__name__) #, static_folder='.'
app._static_folder = os.path.abspath("templates/static/")
CORS(app)


test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

prediction = ''

@app.route('/', methods=['GET'])
def index():
    return render_template('/index.html', title='Learn ASL!')


@app.route('/model', methods = ['POST']) #what is a file path
def predict():
    url = request.form['url']
    response = requests.get(url)
    img = Image.open(StringIO(response.content))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    loaded_model = tf.contrib.saved_model.load_keras_model('./tmp_dir/1556737843/')
    preprocessed_image = applications.mobilenet.preprocess_input(img_array_expanded_dims)
    result = loaded_model.predict(preprocessed_image)
    prediction = labels[np.argmax(result)]

    ret = { 'label': prediction}
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
