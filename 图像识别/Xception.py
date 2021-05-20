import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.data_utils import *
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

'''
basic parameter config
'''
tf.config.experimental.list_physical_devices('GPU')
epochs = 100
batch_size = 64
img_width = 244
img_height = 244

'''
# load the data and define the save dir
'''
data_dir = pathlib.Path(r'/home/zhang.xinxi/CV/data/2000')
save_dir = '/home/zhang.xinxi/CV/checkpoint/finetune_Xception_2000_overfitSolver1/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_dir,
    verbose=1,
    save_weights_only=True,
    period=1)
train_ds, val_ds, class_nums= getdata(dir=data_dir, type='training', batch_size=batch_size, img_width=img_width, img_height=img_height)
train_ds.shuffle(1000)
#train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=1)
print('the data has been loaded')

'''
load the Xception model and to create my own model
'''
# load the Xception_model
Xception_model = keras.applications.Xception(
    weights= 'imagenet',
    input_shape = (img_width, img_height, 3),
    include_top = False,
    classifier_activation=False
)
Xception_model.trainable = True

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
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(class_nums)]
)

# create my own model and compile
inputs = keras.Input(shape=(img_width, img_height, 3))
#normalization
norm_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))

x = data_augmentation(inputs)
x = norm_layer(x)

x = Xception_model(x, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = classifier(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

'''
model training
'''
history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback]
    )

