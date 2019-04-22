import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image
import os

img_width, img_height = 224, 224

train_dir = "/home/ella_feldmann/cv-final/data"
# validation_dir = "/home/ella_feldmann/cv-final/data"

def preprocess_image(file):
    img = image.load_img(train_dir + "/" + file)
    img_array = image.img_to_array(img)
    print(img_array.shape)
    return img_array


base_model = applications.mobilenet.MobileNet(weights='imagenet',include_top=False)

x = base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(2,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)


for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

for filename in os.listdir(train_dir):
    preprocessed_image = preprocess_image(filename)
    result = base_model.predict(preprocessed_image)
    print(results)
