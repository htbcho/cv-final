import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224

train_data_dir = "data/train"
validation_data_dir = "data/val"

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

print(model.summary())
#
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), Activation('relu'), input_shape=(224, 224, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), Activation('relu'), input_shape=(224, 224, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), Activation('relu'), input_shape=(224, 224, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512), Activation('relu'))
# model.add(layers.Dense(5, Activation('softmax')))
