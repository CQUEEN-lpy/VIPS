import numpy as np
import pandas as pd
from urllib.request import urlretrieve
import random
import os
csv_path = r'D:\Users\zhang.xinxi\Desktop\打标\男袖长\阿里\无袖.csv'
target_path = r'D:\Users\zhang.xinxi\Desktop\打标\男袖长\阿里\无袖'

df = np.array(pd.read_csv(csv_path,encoding='GB18030'))
count = 0
for i in df:
    url = i[2]
    count += 1
    id = ''.join(i[0].split())
    s_length = int(len(id)*0.3)
    id = id[:15]

    name = target_path + '\\' + id + '.jpg'
    try:
        urlretrieve(url, name)
    except BaseException as e:
        print(e)
    print(count, id)

print(count)
