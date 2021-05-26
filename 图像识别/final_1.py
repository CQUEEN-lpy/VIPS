'''
import
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import tensorflow as tf
mirrored_strategy = tf.distribute.MirroredStrategy()
import pathlib
from model.myEFF import *

'''
basic parameter config
'''
epochs = 100
batch_size = 128
img_width = 244
img_height = 244
portion = 0.9   # the portion of training set

'''
# load the data and define the save dir and callback function
'''
data_dir = pathlib.Path(r'/home/zhang.xinxi/CV/data/all/train')
save_dir = '/home/zhang.xinxi/CV/checkpoint/test/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_dir,
    verbose=1,
    save_weights_only=True,
    period=1)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=1 - portion,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1 - portion,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

class_nums = class_nums = len(train_ds.class_names)
print('the data has been loaded')

'''
load the EFF model and to create my own model
'''
# learning rate schedule
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

# transfer the label to one_hot form
def map_func(images, label):
    one_hot_label = tf.one_hot(label, depth=class_nums)
    return images, one_hot_label

train_ds = train_ds.map(map_func,num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(map_func,num_parallel_calls=tf.data.AUTOTUNE)

with mirrored_strategy.scope():
    model = get_my_EFFmodel(img_height, img_width, class_nums, '/home/zhang.xinxi/CV/checkpoint/final/cp-0002.ckpt')
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

print(model.summary())

'''
model training
'''
# prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# train
history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback,
                   tf.keras.callbacks.LearningRateScheduler(decay)]
    )
