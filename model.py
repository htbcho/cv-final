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


train_dir = "/home/ella_feldmann/asl_alphabet_train/"
test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def preprocess_image(file):
    img = image.load_img(train_dir + "/" + file)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return applications.mobilenet.preprocess_input(img_array_expanded_dims)

base_model = applications.mobilenet.MobileNet(weights='imagenet',include_top=False)

x = base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(29,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


print("CONSTRUCTED MODEL")

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

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

model.compile(optimizer=tf.train.AdagradOptimizer(0.001), loss='categorical_crossentropy',metrics=['accuracy'])

train_step_size=train_generator.n//train_generator.batch_size
valid_step_size=valid_generator.n//valid_generator.batch_size

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_step_size,
                    validation_data=valid_generator,
                    validation_steps=valid_step_size,
                    epochs=5
)


print("FINISHED TRAINING")

for filename in os.listdir(test_dir):
    img = image.load_img(test_dir + filename)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_image = applications.mobilenet.preprocess_input(img_array_expanded_dims)
    result = model.predict(preprocessed_image)
    print(filename)
    print(np.shape(result))
    print(labels[np.argmax(result)])

output_path = tf.contrib.saved_model.save_keras_model(model, './tmp_dir')
