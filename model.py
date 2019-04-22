import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

img_width, img_height = 224, 224

train_dir = "/home/ella_feldmann/cv-final/data"
# validation_dir = "data/val"


for filename in os.listdir(train_dir):
    print("hello")
    img = image.load_img(train_dir + "/" + filename, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    print(applications.mobilenet.preprocess_input(img_array_expanded_dims))
#     return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
#
# # model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
# base_model = applications.mobilenet.MobileNet(weights='imagenet',include_top=False)
# print(base_model.summary())
#
# x = model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x)
# x=Dense(1024,activation='relu')(x)
# x=Dense(512,activation='relu')(x)
# preds=Dense(2,activation='softmax')(x)
#
# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# step_size_train=train_generator.n//train_generator.batch_size
# model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    epochs=10)
