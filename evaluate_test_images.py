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
from sklearn.metrics import confusion_matrix, accuracy_score



test_dir = "/home/ella_feldmann/asl-alphabet-test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

true_labels = []
pred_labels = []

loaded_model = tf.contrib.saved_model.load_keras_model('./scratch_tmp_dir/1556848947/') # SCRATCH MODEL ONLY !!!
# loaded_model = tf.contrib.saved_model.load_keras_model('./tmp_dir/1556737843/') # MOBILENET ONLY !!!!

for subdir in os.listdir(test_dir):

    for filename in os.listdir(test_dir + subdir):
        if (filename != ".DS_Store"):
            test_image = image.load_img(test_dir + subdir + '/' + filename, target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            # test_image = applications.mobilenet.preprocess_input(test_image) # MOBILENET ONLY !!!!!

            result = loaded_model.predict(test_image)
            true_labels.append(subdir) # True labels
            pred_labels.append(labels[np.argmax(result)]) # Model predictions


confusion = confusion_matrix(true_labels, pred_labels, labels)
print(confusion)

plt.figure(0, figsize =(7,7))
plt.imshow(confusion, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.title('Confusion Matrix without Normalization')
plt.xlabel('Predicted Label', fontsize = 16)
plt.ylabel('True Label', fontsize = 16)
plt.xticks(np.arange(29), labels)
plt.yticks(np.arange(29), labels)
plt.colorbar()
thresh = confusion.max() / 2.
plt.savefig('confusion_matrix.png')



plt.figure(1, figsize =(7,7))
norm_confusion = confusion.astype('float') / confusion.sum(axis=1)[:,np.newaxis]
plt.imshow(norm_confusion, interpolation = 'nearest', cmap = plt.cm.Greens)
plt.xlabel('Predicted Label', fontsize = 16)
plt.ylabel('True Label', fontsize = 16)
plt.xticks( np.arange(26), labels)
plt.yticks( np.arange(26), labels)
plt.colorbar()
plt.title('Confusion Matrix with Normalization')
thresh = norm_confusion.max() / 2.

plt.savefig('norm_confusion_matrix.png')

print(true_labels[0:10])
print(pred_labels[0:10])
acc = accuracy_score(true_labels, pred_labels)
print(acc)
