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


train_dir = "/home/ella_feldmann/asl_alphabet_train/"
test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def preprocess_image(img):
    img = image.load_img(train_dir + "/" + img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# dimensions of our images.
img_width, img_height = 224, 224

input_tensor = Input(shape=(224, 224, 3))
vgg_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=input_tensor)

flatten = Flatten()
x = flatten(model.output)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
x = Dense(29, activation='softmax')(x)
model = Model(inputs=vgg_model.input, outputs=x)

for layer in model.layers:
layer.trainable=False
for layer in model.layers[-2:]:
layer.trainable=True

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

 # fit_model(model, batches, val_batches, 2)



# # Getting output tensor of the last VGG layer that we want to include
# # x = layer_dict['block2_pool'].output
# x = vgg_model.output
# # Stacking a new simple convolutional network on top of it
# # x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(29, activation='softmax')(x)
#
# custom_model=Model(inputs=vgg_model.input,outputs=x)
#
# for layer in custom_model.layers[:7]:
#     layer.trainable = False
# for layer in custom_model.layers[7:]:
#     layer.trainable = True
#
# # custom_model.compile(loss='categorical_crossentropy',
# #                      optimizer='rmsprop',
# #                      metrics=['accuracy'])


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

#
# for subdir in os.listdir(test_dir):
#
#     for filename in os.listdir(test_dir + subdir):
#
#         test_image = image.load_img(test_dir + subdir + '/' + filename, target_size = (224, 224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         # test_image = applications.mobilenet.preprocess_input(test_image) # MOBILENET ONLY !!!!!
#
#         result = loaded_model.predict(test_image)
#         true_labels.append(subdir) # True labels
#         pred_labels.append(labels[np.argmax(result)]) # Model predictions
#
#
# confusion = confusion_matrix(true_labels, pred_labels, labels)
# print(confusion)
#
# plt.figure(0, figsize =(7,7))
# plt.imshow(confusion, interpolation = 'nearest', cmap = plt.cm.Blues)
# plt.title('Confusion Matrix without Normalization')
# plt.xlabel('Predicted Label', fontsize = 16)
# plt.ylabel('True Label', fontsize = 16)
# plt.xticks(np.arange(29), labels)
# plt.yticks(np.arange(29), labels)
# plt.colorbar()
# thresh = confusion.max() / 2.
# plt.savefig('confusion_matrix.png')
#
#
#
# plt.figure(1, figsize =(7,7))
# norm_confusion = confusion.astype('float') / confusion.sum(axis=1)[:,np.newaxis]
# plt.imshow(norm_confusion, interpolation = 'nearest', cmap = plt.cm.Greens)
# plt.xlabel('Predicted Label', fontsize = 16)
# plt.ylabel('True Label', fontsize = 16)
# plt.xticks( np.arange(29), labels)
# plt.yticks( np.arange(29), labels)
# plt.colorbar()
# plt.title('Confusion Matrix with Normalization')
# thresh = norm_confusion.max() / 2.
#
# plt.savefig('norm_confusion_matrix.png')
#
# print(true_labels[0:100])
# print(pred_labels[0:100])
# acc = accuracy_score(true_labels, pred_labels)
# print(acc)
# #
# # output_path = tf.contrib.saved_model.save_keras_model(model, './tmp_dir')
# # print("SAVED AT " + output_path)
