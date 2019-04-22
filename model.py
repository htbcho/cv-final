import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('hello I am working')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), layers.Activation('relu'), input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), layers.Activation('relu'), input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), layers.Activation('relu'), input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512), layers.Activation('relu'))
model.add(layers.Dense(5, layers.Activation('softmax')))
