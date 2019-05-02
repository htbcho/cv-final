import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import os
from tensorflow.keras.models import load_model

test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def predict(img):

    loaded_model = tf.contrib.saved_model.load_keras_model('./tmp_dir/1556737843/')

    # for filename in os.listdir(test_dir):
    # img = image.load_img(test_dir + filename)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_image = applications.mobilenet.preprocess_input(img_array_expanded_dims)
    result = loaded_model.predict(preprocessed_image)
    print(filename)
    print(np.shape(result))
    print(labels[np.argmax(result)])
    return labels[np.argmax(result)]
