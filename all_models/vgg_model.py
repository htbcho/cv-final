import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D, Input, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils.vis_utils import plot_model



train_dir = "/home/ella_feldmann/asl_alphabet_train/"
train_dir_2 = "/home/ella_feldmann/cv-final/test/"
test_dir = "/home/ella_feldmann/asl-alphabet-test/"

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

# COMPILE THE DENSE LAYER MODEL
vgg_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(224, 224, 3))


model = models.Sequential()
model.add(vgg_model)
model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(29, activation='softmax'))

model.summary()
vgg_model.trainable = False

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2, rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                 subset = 'training',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=20, # total number of training images should be divisible by batch size
                                                 class_mode='categorical',
                                                 shuffle=True)

valid_generator = train_datagen.flow_from_directory(train_dir,
                                                 subset = 'validation',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=20, # total number of training images should be divisible by batch size
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.AdagradOptimizer(0.001),
              metrics=["accuracy"])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=22,
      validation_data=valid_generator,
      validation_steps=50,
      verbose=2)



model.save('vgg_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
loaded_model = load_model('vgg_model.h5')
true_labels = []
pred_labels = []

for subdir in os.listdir(test_dir):
    for filename in os.listdir(test_dir + subdir):

        test_image = image.load_img(test_dir + subdir + '/' + filename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.divide(test_image, 255.0)
        test_image = np.expand_dims(test_image, axis = 0)
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
plt.xticks( np.arange(29), labels)
plt.yticks( np.arange(29), labels)
plt.colorbar()
plt.title('Confusion Matrix with Normalization')
thresh = norm_confusion.max() / 2.

plt.savefig('norm_confusion_matrix.png')

print(true_labels[0:100])
print(pred_labels[0:100])
acc = accuracy_score(true_labels, pred_labels)
print(acc)
