import os
import numpy as np
import pandas as pd
def get_consistancy(l1,l2):
    table = {}
    for i in l1:
        table[i] = 1
    intersection = 0
    union = len(l1)
    for j in l2:
        if j in table:
            intersection += 1
        else:
            union += 1
    if union == 0:
        proportion = -1
    else:
        proportion = intersection/union
    return proportion, intersection, union

def func1(path1, path2, bigtype):
    tmp_list1 = os.listdir(path1)
    tmp_list2 = os.listdir(path2)
    for i in tmp_list1:
        if '.' in i or i not in tmp_list2:
            continue
        tp1 = path1 + '\\' + i
        print(tp1)
        if "新建文件夹" in tp1:
            print(1)
        tp2 = path2 + '\\' + i
        print(tp2)
        proportion, intersection, union= get_consistancy(os.listdir(tp1), os.listdir(tp2))
        print(proportion)
        DF.append([bigtype, i, proportion, intersection, union])

filepath1 = r'Y:\运营中心公共文件夹\NLP\连衣裙\黄芳'
list_people_1 = os.listdir(filepath1)
filepath2 = r'Y:\运营中心公共文件夹\NLP\连衣裙\邬璇'
list_people_2 = os.listdir(filepath2)

DF = []
for i in list_people_1:
    if '.' in i:
        continue
    tmp_path_1 = filepath1 + '\\' + i
    tmp_path_2 = filepath2 + '\\' + i
    func1(tmp_path_1, tmp_path_2, i)
DF = pd.DataFrame(DF)

DF.columns = ['大类', '小类', '一致率', 'intersection', 'union']
DF = DF.set_index('大类')
DF.to_csv(r'D:\Users\zhang.xinxi\Desktop\新建文件夹\1.csv', encoding='utf_8_sig')
print(DF)
