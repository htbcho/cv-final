import numpy as np
import os
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
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


train_dir = "/home/ella_feldmann/asl_alphabet_train/"
test_dir = "/home/ella_feldmann/asl_alphabet_test/"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]


base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
# x = Dropout(0.4)(x)
predictions = Dense(29, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers: # first: train only the top layers (which were randomly initialized)
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy' ,metrics=['accuracy'])

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


train_step_size=train_generator.n//train_generator.batch_size
valid_step_size=valid_generator.n//valid_generator.batch_size

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_step_size,
                    validation_data=valid_generator,
                    validation_steps=valid_step_size,
                    epochs=3
)

# let's visualize layer names and layer indices to see how many layers we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze  the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_step_size,
                    validation_data=valid_generator,
                    validation_steps=valid_step_size,
                    epochs=3
)



output_path = tf.contrib.saved_model.save_keras_model(model, './inc_tmp_dir')

# loaded_model = tf.contrib.saved_model.load_keras_model('./scratch_tmp_dir/1556986629/') # SCRATCH MODEL ONLY !!!

true_labels = []
pred_labels = []


for subdir in os.listdir(test_dir):
    for filename in os.listdir(test_dir + subdir):

        test_image = image.load_img(test_dir + subdir + '/' + filename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = preprocess_input(test_image)

        result = model.predict(test_image)
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
#
# output_path = tf.contrib.saved_model.save_keras_model(model, './tmp_dir')
# print("SAVED AT " + output_path)
