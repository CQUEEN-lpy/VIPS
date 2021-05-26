'''
import========================================================================================================================================
'''
import os
import pandas
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from model.myEFF_predict import *
from PIL import Image
import numpy as np

'''
basic parameter config========================================================================================================================
'''
epochs = 100
batch_size = 256
img_width = 244
img_height = 244
class_nums = 4
checkpoint_path = '/home/zhang.xinxi/CV/checkpoint/test/cp-0008.ckpt'
filePath = '/home/zhang.xinxi/CV/data/all2/all/2'
savePath = '/home/zhang.xinxi/CV/data/all2/csv/final'
savePath += '/02.csv'
class_names = ['V领', '圆领', '方领', '一字领']

"""
build the model and load it from the check point =============================================================================================
"""
model = get_my_EFFmodel(img_height, img_width, class_nums, checkpoint_path)

"""
predict======================================================================================================================================
"""
result = []

file_names = os.listdir(filePath)

j = 1               # batch index
images = []         # image column
paths = []          # path column
classes = []        # class column
final = None        # the final pandas
count = 0           # count the number of image
for i in file_names:
    count += 1
    j += 1

    try:
        if j > batch_size:
            print(count)
            images = tf.convert_to_tensor(images, dtype='float32')
            outputs = np.array(model(images))
            index = np.array(tf.math.argmax(outputs, 1))
            scores = []
            for k in range(index.size):
                scores.append(outputs[k][index[k]])
                classes.append(class_names[index[k]])
            index = np.expand_dims(index, 1)
            scores = np.expand_dims(np.array(scores), 1)
            classes = np.expand_dims(np.array(classes), 1)
            paths = np.expand_dims(np.array(paths), 1)
            total = np.concatenate((paths, index), 1)
            total = np.concatenate((total, scores), 1)
            total = np.concatenate((total, classes), 1)
            if final is not None:
                final = np.concatenate((final, total), 0)
            else:
                final = total
            paths = []
            images = []
            classes = []
            j = 1
    except BaseException as ex:
        print(ex)
        print('111111')
        paths = []
        images = []
        classes = []
        j = 1

    try:
        img = np.array(Image.open(filePath + '/' + i))
        img = tf.image.resize(img, (244, 244))

        if img.shape == (244, 244, 3):
            paths.append(i)
            images.append(img)
    except BaseException as ex:
        print(
            '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(ex)
        print(i)
        print(img)

print(count)
images = tf.convert_to_tensor(images, dtype='float32')
outputs = np.array(model(images))
index = np.array(tf.math.argmax(outputs, 1))
scores = []
for k in range(index.size):
    scores.append(outputs[k][index[k]])
    classes.append(class_names[index[k]])

index = np.expand_dims(index, 1)
scores = np.expand_dims(np.array(scores), 1)
classes = np.expand_dims(np.array(classes), 1)
paths = np.expand_dims(np.array(paths), 1)

total = np.concatenate((paths, index), 1)
total = np.concatenate((total, scores), 1)
total = np.concatenate((total, classes), 1)
if final is not None:
    final = np.concatenate((final, total), 0)
else:
    final = total

final = pandas.DataFrame(final)
final.columns = ['文件名', '类别id', '得分', '类别']
final = final.set_index('文件名')
final.to_csv(savePath, encoding='utf_8_sig')
