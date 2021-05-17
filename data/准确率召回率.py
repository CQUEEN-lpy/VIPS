import pandas as pd
import os
import numpy as np
DF = pd.read_excel(r'D:\Users\zhang.xinxi\Desktop\新建文件夹\分词.xlsx')
print(DF)

def func1(path1, path2):
    tmp_list1 = os.listdir(path1)
    tmp_list2 = os.listdir(path2)
    for i in tmp_list1:
        if '.' in i or i not in tmp_list2:
            continue
        tp1 = path1 + '\\' + i
        tp2 = path2 + '\\' + i
        build_table(os.listdir(tp1), os.listdir(tp2),i)

def build_table(l1,l2,type):
    for i in l1:
        tmp = i[:-4]
        if type not in table_true:
            table_true[type] = [tmp]
        else:
            table_true[type].append(tmp)




filepath1 = r'Y:\运营中心公共文件夹\NLP\连衣裙\黄芳'
list_people_1 = os.listdir(filepath1)
filepath2 = r'Y:\运营中心公共文件夹\NLP\连衣裙\邬璇'
list_people_2 = os.listdir(filepath2)
table_true = {}

for i in list_people_1:
    if '.' in i:
        continue
    tmp_path_1 = filepath1 + '\\' + i
    tmp_path_2 = filepath2 + '\\' + i
    func1(tmp_path_1, tmp_path_2)

fenci_list = np.array(DF)
TP_table = {}
test_table = {}

for i in fenci_list:
    type = i[1]
    id = str(i[0])
    if type in table_true:
        if id in table_true[type]:
            if type not in TP_table:
                TP_table[type] = 1
            else:
                TP_table[type] += 1
    if type not in test_table:
        test_table[type] = 1
    else:
        test_table[type] += 1

l = []
for i in test_table:
    total_true = 0
    if i not in TP_table:
        true_num = 0
        continue
    else:
        total_true = len(table_true[i])
        true_num = TP_table[i]
    if total_true == 0:
        zhaohui = 0
    else:
        zhaohui = true_num/total_true
    l.append([i, true_num, test_table[i], total_true, true_num/test_table[i], zhaohui])
DF = pd.DataFrame(l)
DF.columns = ['标签', '正确个数', '分词集中所含商品个数', '人工打标签所含商品个数', '正确率', '召回率']
DF = DF.set_index('标签')
DF.to_csv(r'D:\Users\zhang.xinxi\Desktop\新建文件夹\分词一致性判断.csv', encoding='utf_8_sig')


