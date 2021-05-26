import tensorflow as tf
from tensorflow import keras
from utils.Grid_Mask import GridMask
import tensorflow.keras.layers as layers
def get_my_EFFmodel(img_height, img_width, class_nums, checkpoint=None):
    # load the EFF_model
    if checkpoint is None:
        EFF_model = tf.keras.applications.EfficientNetB6(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(img_height, img_width, 3), pooling=None, classes=class_nums,
            classifier_activation=None,
        )
    else:
        EFF_model = tf.keras.applications.EfficientNetB6(
            include_top=False, weights=None, input_tensor=None,
            input_shape=(img_height, img_width, 3), pooling=None, classes=class_nums,
            classifier_activation=None,
        )
    EFF_model.trainable = True

    data_augmentation = GridMask(200, 300)

    # classifier
    classifier = keras.Sequential([
        layers.Dropout(0.5),
        layers.Dense(class_nums, activation='softmax')]
    )

    # normalization
    norm_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))

    # create my own model and compile
    inputs = keras.Input(shape=(img_width, img_height, 3))

    x = data_augmentation(inputs)
    x = norm_layer(x)
    x = EFF_model(x, training=True)
    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = classifier(x)
    model = keras.Model(inputs, outputs)
    if checkpoint is not None:
        model.load_weights(checkpoint)
    return model
