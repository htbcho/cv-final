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

test_dir = "/home/ella_feldmann/cv-final/test/A/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]


loaded_model = tf.contrib.saved_model.load_keras_model('./scratch_tmp_dir/1556848947/')

for filename in os.listdir(test_dir):
    # test_image = image.load_img(test_dir + filename, target_size = (64, 64))
    # test_image = image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis = 0)
    #
    # #predict the result
    # result = loaded_model.predict(test_image)
    print(filename)
    # print(labels[np.argmax(result)])
