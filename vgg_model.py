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

train_dir = "/home/ella_feldmann/asl_alphabet_train/"
test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def preprocess_image(img):
    print(img.shape)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims

# dimensions of our images.
img_width, img_height = 224, 224

input_tensor = Input(shape=(224, 224, 3))
vgg_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=input_tensor)

# Creating dictionary that maps layer names to the layers
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# Getting output tensor of the last VGG layer that we want to include
# x = layer_dict['block2_pool'].output
x = vgg_model.output
# Stacking a new simple convolutional network on top of it
# x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(29, activation='softmax')(x)

custom_model=Model(inputs=vgg_model.input,outputs=x)

for layer in custom_model.layers[:7]:
    layer.trainable = False


# custom_model.compile(loss='categorical_crossentropy',
#                      optimizer='rmsprop',
#                      metrics=['accuracy'])

custom_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_image, validation_split=0.2)

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

train_step_size=train_generator.n//train_generator.batch_size
valid_step_size=valid_generator.n//valid_generator.batch_size

history = custom_model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_step_size,
                    validation_data=valid_generator,
                    validation_steps=valid_step_size,
                    epochs=5
)


print("FINISHED TRAINING")

for filename in os.listdir(test_dir):
    img = preprocess_input(filename)
    result = custom_model.predict(img)
    print(filename)
    print(labels[np.argmax(result)])

output_path = tf.contrib.saved_model.save_keras_model(model, './tmp_dir')
print("SAVED AT " + output_path)
