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
from keras.preprocessing.image import ImageDataGenerator



train_dir = "/home/ella_feldmann/asl_alphabet_train/"
# test_dir = "/home/ella_feldmann/asl-alphabet-test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

# GATHER ALL THE TRAINING, TESTING, AND VALIDATION DATA


# COMPILE THE DENSE LAYER MODEL
vgg_model = VGG16(weights='imagenet',
                   include_top=False)


model = models.Sequential()
model.add(vgg_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(29, activation='softmax'))

model.summary()
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

vgg_model.trainable = False

print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])



train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
#
# #
# # from keras import models
# # from keras import layers
# # from keras import optimizers
# #
# # model = models.Sequential()
# # model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
# # model.add(layers.Dropout(0.5))
# # model.add(layers.Dense(1, activation='sigmoid'))
# #
# # model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
# #               loss='binary_crossentropy',
# #               metrics=['acc'])
# #
# # history = model.fit(train_features, train_labels,
# #                     epochs=30,
# #                     batch_size=20,
# #                     validation_data=(validation_features, validation_labels))
#
#
# for layer in model.layers:
#     layer.trainable=False
# for layer in model.layers[-2:]:
#     layer.trainable=True
#
# opt = RMSprop(lr=0.001)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#
#  # fit_model(model, batches, val_batches, 2)
#
#
#
# # # Getting output tensor of the last VGG layer that we want to include
# # # x = layer_dict['block2_pool'].output
# # x = vgg_model.output
# # # Stacking a new simple convolutional network on top of it
# # # x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
# # x = MaxPooling2D(pool_size=(2, 2))(x)
# # x = Flatten()(x)
# # x = Dense(256, activation='relu')(x)
# # x = Dropout(0.5)(x)
# # x = Dense(29, activation='softmax')(x)
# #
# # custom_model=Model(inputs=vgg_model.input,outputs=x)
# #
# # for layer in custom_model.layers[:7]:
# #     layer.trainable = False
# # for layer in custom_model.layers[7:]:
# #     layer.trainable = True
# #
# # # custom_model.compile(loss='categorical_crossentropy',
# # #                      optimizer='rmsprop',
# # #                      metrics=['accuracy'])
#
#
# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
#
# train_generator = train_datagen.flow_from_directory(train_dir,
#                                                  subset = 'training',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=20, # total number of training images should be divisible by batch size
#                                                  class_mode='categorical',
#                                                  shuffle=True)
#
# valid_generator = train_datagen.flow_from_directory(train_dir,
#                                                  subset = 'validation',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=20, # total number of training images should be divisible by batch size
#                                                  class_mode='categorical',
#                                                  shuffle=True)
#
#
# train_step_size=train_generator.n//train_generator.batch_size
# valid_step_size=valid_generator.n//valid_generator.batch_size
#
# history = model.fit_generator(generator=train_generator,
#                     steps_per_epoch=train_step_size,
#                     validation_data=valid_generator,
#                     validation_steps=valid_step_size,
#                     epochs=5
# )
#
#
# def preprocess_image(img):
#     img = image.load_img(train_dir + "/" + img)
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     return img
# #
# # for subdir in os.listdir(test_dir):
# #
# #     for filename in os.listdir(test_dir + subdir):
# #
# #         test_image = image.load_img(test_dir + subdir + '/' + filename, target_size = (224, 224))
# #         test_image = image.img_to_array(test_image)
# #         test_image = np.expand_dims(test_image, axis = 0)
# #         # test_image = applications.mobilenet.preprocess_input(test_image) # MOBILENET ONLY !!!!!
# #
# #         result = loaded_model.predict(test_image)
# #         true_labels.append(subdir) # True labels
# #         pred_labels.append(labels[np.argmax(result)]) # Model predictions
# #
# #
# # confusion = confusion_matrix(true_labels, pred_labels, labels)
# # print(confusion)
# #
# # plt.figure(0, figsize =(7,7))
# # plt.imshow(confusion, interpolation = 'nearest', cmap = plt.cm.Blues)
# # plt.title('Confusion Matrix without Normalization')
# # plt.xlabel('Predicted Label', fontsize = 16)
# # plt.ylabel('True Label', fontsize = 16)
# # plt.xticks(np.arange(29), labels)
# # plt.yticks(np.arange(29), labels)
# # plt.colorbar()
# # thresh = confusion.max() / 2.
# # plt.savefig('confusion_matrix.png')
# #
# #
# #
# # plt.figure(1, figsize =(7,7))
# # norm_confusion = confusion.astype('float') / confusion.sum(axis=1)[:,np.newaxis]
# # plt.imshow(norm_confusion, interpolation = 'nearest', cmap = plt.cm.Greens)
# # plt.xlabel('Predicted Label', fontsize = 16)
# # plt.ylabel('True Label', fontsize = 16)
# # plt.xticks( np.arange(29), labels)
# # plt.yticks( np.arange(29), labels)
# # plt.colorbar()
# # plt.title('Confusion Matrix with Normalization')
# # thresh = norm_confusion.max() / 2.
# #
# # plt.savefig('norm_confusion_matrix.png')
# #
# # print(true_labels[0:100])
# # print(pred_labels[0:100])
# # acc = accuracy_score(true_labels, pred_labels)
# # print(acc)
# # #
# # # output_path = tf.contrib.saved_model.save_keras_model(model, './tmp_dir')
# # # print("SAVED AT " + output_path)
