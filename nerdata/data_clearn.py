# Coding by long
# Datatime:2022/3/1 15:51
# Filename:data_clearn.py
# Toolby: PyCharm
# ______________coding_____________
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
def clearndata():
    data_frame = pd.read_csv('维族人名.csv', sep='\t',header=None)
    print(data_frame.columns)
    list_name_all = []
    for i in data_frame[0]:
        list_name = i.replace(' ', '').replace('，', ',').replace(')',
                                                                 '').replace('(', ',').replace('"','').replace(';',',').split(sep=',')
        print(list_name)
        list_name_all.extend(list_name)

    data = {'name': list_name_all}
    dataframe1 = pd.DataFrame(data)
    dataframe1.to_csv('维族人名切分.csv', encoding='utf-8')


def get_name(filepath):
    """
    :param filepath: 文件的读取路径
    :return: 0
    """
    f_name = open('name_clean_peopledaily.txt', 'w', encoding='utf-8')
    with open(filepath, 'r', encoding='utf-8') as file:
        files = file.readlines()
        for i in files:
            data_name = {}
            entity_list_name = []
            data = eval(i)
            text = data['text']
            entity_list = data['entity_list']
            for i in entity_list:
                if i['entity_type'] == "PER" or i['entity_type'] == 'NAME' or i['entity_type'] == 'PERSON':  # "NAME"
                    entity_list_name.append(i)
            if entity_list_name:  # 如果存在元素,有元素即为真
                data_name['text'] = text
                data_name['entity_list'] = entity_list_name
            if data_name:  # 如果生成的名称数据不为空，则往里面添加相关的元素
                f_name.write(str(data_name)+'\n')
    f_name.close()





            #print(i, type(i))



# get_name('./open_ner_data/people_daily/people_daily_ner.txt')

# 新疆维吾尔族人名替换
import random
def concat():
    """
    :return:
    """
    random.seed(666)  # 定义随机数种子
    data_frame = pd.read_csv('维族人名切分.csv', sep=',', encoding='utf-8')
    name = data_frame['name'].values.tolist()
    # print(name,len(name))
    index1 = [index for index in range(len(name))]
    index2 = [index for index in range(len(name))]
    random.shuffle(index1)
    random.shuffle(index2)  # 将里面的名字全部打乱，用于随机生成维吾尔族人名，但得记得，人名应该有相关的规则，比如中间是使用“·”进行分割
    # 随机生成3万个名字
    # 根据两份文件随机生成三万个名字
    name_all = []
    for i, j in zip(index1, index2):
        name_here1 = str(name[i]) + '·' + str(name[j])
        if name_here1!='':
            name_all.append(name_here1)  # 随机拼一些名字，名字的个数为len()
        x = len(index1)-i-1
        y = len(index1)-j-1
        name_here2 = str(name[x]) + '·' + str(name[y])
        # print(name_here2)
        if name_here2 != '':
            name_all.append(name_here2)
    return name_all


def read_name(file_path, num):
    """

    :param file_path: 读取名字的地址
    :param num: 随机读取名字的个数
    :return: list列表
    """
    random.seed(666)
    dataframe = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    name = dataframe.name.values.tolist()
    random.shuffle(name)
    if num>30000 or num<0:
        print('请重新运行程序，名字数量有误')
        name_here = []
    else:
        name_here = name[:num]
        # print(name_here,len(name_here))
    return name_here



def repalce_name_entity(fileoutput,fileinput,replace_persent):
    """

    :param fileoutput: 文件输出路径
    :param fileinput:文件输入路径
    :param replace_persent:名字被替换的概率，即文本中各种名字被mask掉的比率。
    :return:文档文件，还不需要进行标签的制作
    """
    random.seed(666) # 定义随机数种子
    data_frame = pd.read_csv('维族人名切分.csv', sep=',', encoding='utf-8')
    name = data_frame['name'].values.tolist()
    # print(name,len(name))
    index1 = [index for index in range(len(name))]
    index2 = [index for index in range(len(name))]
    random.shuffle(index1)
    random.shuffle(index2)   # 将里面的名字全部打乱，用于随机生成维吾尔族人名，但得记得，人名应该有相关的规则，比如中间是使用“·”进行分割
    # 随机生成3万个名字
    # 根据两份文件随机生成三万个名字
    name_concat = concat()
    name_alreadly = read_name('name.csv', 10000)  # 从已有的名字中取一部分名字
    name_concat.extend(name_alreadly)
    name_all = name_concat
    print(len(name_all))
    str1 = "outreplace_name"+str(replace_persent)+'.txt'
    f_out = open(str1,"w",encoding='utf-8')
    with open('name_clean.txt', "r", encoding='utf-8') as file:
        datalines = file.readlines()   # 至此文档中的文字已经全部读取
        data_len = len(datalines)
        random.shuffle(datalines)
        data_replace_len = int(data_len * replace_persent)
        print(data_replace_len)
        for index,i in zip(range(data_replace_len),datalines[:data_replace_len]):
            i = eval(i)  # 此时的数据已经拿到
            # print(i)
            text = i['text']
            # print(text)
            entity_list = i['entity_list']
            count = 0
            entity_list_new = []
            for i in entity_list:
                entity_index = {}
                entity_type = "NAME"
                entity_name_new = name_all[index]
                rentity_len = len(entity_name_new)
                text = text.replace(i['entity'], entity_name_new)
                begin = i['entity_index']['begin'] +count
                end = i['entity_index']['end'] + rentity_len - len(i['entity']) + count
                ent_message = {}
                entity_index['begin'] = begin
                entity_index['end'] = end
                ent_message['entity_index'] = entity_index
                ent_message['entity_type'] = 'NAME'
                ent_message['entity'] = entity_name_new
                index = random.randint(0, 10000)
                entity_list_new.append(ent_message)
                count += rentity_len - len(i['entity'])   # 累计计算相关的下标
            data_each = {}
            data_each['text'] = text
            data_each['entity_list'] = entity_list_new
            f_out.write(str(data_each)+'\n')
            # 然后还需要把剩下的人名数据写到训练集中
        for i in datalines[data_replace_len:]:
            f_out.write(str(i))
    f_out.close()
    return 0


repalce_name_entity(1,1,0.8)










