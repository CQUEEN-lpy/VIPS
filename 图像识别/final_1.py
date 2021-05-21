# training acc=90; eval acc=75
'''
import
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import pathlib
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


'''
basic parameter config
'''
epochs = 100
batch_size = 4
img_width = 500
img_height = 500
portion = 0.8   # the portion of training set

'''
# load the data and define the save dir and callback function
'''
data_dir = pathlib.Path(r'/home/zhang.xinxi/CV/data/image2/part')
save_dir = '/home/zhang.xinxi/CV/checkpoint/big/cp-{epoch:04d}.ckpt'
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

# load the EFF_model
EFF_model = tf.keras.applications.EfficientNetB6(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(img_height, img_width, 3), pooling=None, classes=class_nums,
    classifier_activation=None,
)
EFF_model.trainable = True

# data augmentation
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# classifier
classifier = keras.Sequential([
    layers.Dropout(0.5),
    layers.Dense(class_nums, activation='softmax')]
)

# normalization
norm_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))

# transfer the label to one_hot form
def map_func(images, label):
  one_hot_label = tf.one_hot(label, depth=class_nums)
  return images, one_hot_label

train_ds = train_ds.map(map_func)
val_ds = val_ds.map(map_func)

# check the data
for i, j in train_ds.take(1):
    print(j)

# create my own model and compile
inputs = keras.Input(shape=(img_width, img_height, 3))

x = data_augmentation(inputs)
x = norm_layer(x)
x = EFF_model(x, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)

outputs = classifier(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

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
