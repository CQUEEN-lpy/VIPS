import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.data_utils import *
import tensorflow.keras as keras
import numpy as np

'''
basic parameter config
'''
tf.config.experimental.list_physical_devices('GPU')
epochs = 10
batch_size = 32
img_width = 1200
img_height = 1200

'''
# load the data and define the save dir
'''
data_dir = pathlib.Path(r'/home/zhang.xinxi/CV/data/image1')
save_dir = '/home/zhang.xinxi/CV/checkpoint/Xception/cp-{epoch:04d}.ckpt'
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
    input_shape = (1200,1200,3),
    include_top = False,
    classifier_activation=False
)
Xception_model.trainable = False

# create my own model and compile
inputs = keras.Input(shape=(1200, 1200, 3))
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
x = norm_layer(inputs)
norm_layer.set_weights([mean, var])
x = Xception_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(class_nums)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())
history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=epochs,
        callbacks=[cp_callback]
    )

