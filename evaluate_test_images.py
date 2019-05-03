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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



test_dir = "/home/ella_feldmann/cv-final/test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

true_labels = []
pred_labels = []

loaded_model = tf.contrib.saved_model.load_keras_model('./scratch_tmp_dir/1556848947/')

for subdir in os.listdir(test_dir):

    for filename in os.listdir(test_dir + subdir):

        test_image = image.load_img(test_dir + subdir + '/' + filename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)

        true_labels.append(subdir) # True labels
        pred_labels.append(labels[np.argmax(result)]) # Model predictions


    confusion = confusion_matrix(ture_labels, pred_labels)

    plt.figure(0, figsize =(7,7))
    plt.imshow(confusion, interpolation = 'nearest', cmap = plt.cm.Blues)
    classes = ['blowdown', 'other', 'forest', 'cloud', 'blooming']
    plt.title('Confusion Matrix without Normalization')
    plt.xlabel('Predicted Label', fontsize = 16)
    plt.ylabel('True Label', fontsize = 16)
    plt.xticks( np.arange(NUM_CLASSES), (label_map.keys()))
    plt.yticks( np.arange(NUM_CLASSES), (label_map.keys()))
    plt.colorbar()
    thresh = confusion.max() / 2.
    plt.savefig('confusion_matrix.png')
