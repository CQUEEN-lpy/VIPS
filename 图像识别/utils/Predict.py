'''
import========================================================================================================================================
'''
import os

import pandas

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pathlib
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from PIL import Image
import numpy as np

'''
basic parameter config=======================================================================================================================================
'''
epochs = 100
batch_size = 64
img_width = 244
img_height = 244
class_nums = 4
checkpoint_path = '/home/zhang.xinxi/CV/checkpoint/mask_all/cp-0003.ckpt'
data_path = pathlib.Path(r'/home/zhang.xinxi/CV/data/all')

# load the data
ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

"""
build the model and load it from the check point =======================================================================================================================================
"""
# load the EFF_model
EFF_model = tf.keras.applications.EfficientNetB6(
    include_top=False, weights=None, input_tensor=None,
    input_shape=(img_height, img_width, 3), pooling=None, classes=class_nums,
    classifier_activation=None,
)
EFF_model.trainable = False


# classifier
classifier = keras.Sequential([
    #layers.Dropout(0.5),
    layers.Dense(class_nums, activation='softmax')]
)

# normalization
norm_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))


# create my own model and compile
inputs = keras.Input(shape=(img_width, img_height, 3))
x = inputs
x = norm_layer(x)
x = EFF_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = classifier(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])
#model.load_weights(checkpoint_path)

"""
predict====================================================================================================================
"""
result = []
filePath = '/home/zhang.xinxi/CV/data/all/0'
file_names = os.listdir(filePath)
j = 1
images = []
paths = []
final = None
count = 0
for i in file_names:
    count += 1
    if j <= batch_size:
        paths.append(i)
        img = np.array(Image.open(filePath + '/' +i))
        img = tf.image.resize(img, (244, 244))
        j += 1
        images.append(img)
    else:
        print(count)
        images = tf.convert_to_tensor(images, dtype='float32')
        outputs = np.array(model(images))
        index = np.array(tf.math.argmax(outputs, 1))
        scores = []
        for k in range(index.size):
            scores.append(outputs[k][index[k]])
        index = np.expand_dims(index, 1)
        scores = np.expand_dims(np.array(scores), 1)
        paths = np.expand_dims(np.array(paths), 1)
        total = np.concatenate((paths, index), 1)
        total = np.concatenate((total, scores), 1)
        if final is not None:
            final = np.concatenate((final, total), 0)
        else:
            final = total
        paths = []
        images = []
        j = 1

save_dir = ''
final = pandas.DataFrame(final)
final.set_index(0)
final.to_csv(save_dir)
