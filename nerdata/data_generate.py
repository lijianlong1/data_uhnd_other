# Coding by long
# Datatime:2022/4/28 12:43
# Filename:data_generate.py
# Toolby: PyCharm
# description:
# ______________coding_____________
labels = {'O': 0, 'B_NAME': 1, 'I_NAME': 2, }   # 目前使用粗粒度进行相关的标记,直接使用相关的细粒度方式对文本进行标注。
# 其中unknow表示，未知实体，用作保留字段
# B_Name表示名字的开头
from tqdm import tqdm
# 用于记录有多少条数据是可以被单独编码的数据
dataname_out = open('data_out_per_word.txt', "w", encoding='utf-8')
with open('outreplace_name0.8.txt','r',encoding='utf-8') as f:
    data_lines = f.readlines()[:]
    for i, j in zip(data_lines[:], tqdm(range(len(data_lines[:])))):
        i = eval(i)
        text = i['text']
        text_label = ['O' for label in range(len(text)+200)]  # 先开辟一个大型的数组，保证长度不会溢出。加上一个200的缓冲空间
        entity_list = i['entity_list']
        for message in entity_list:
            entity_index = message['entity_index']  # 此时已经拿到了相应的entity标签
            # #########################在需要对实体进行识别的情况下，加入实体标签判别函数，用于对实体值进行判别，用于制作实体标签###############################################
            begin = entity_index['begin']
            end = entity_index['end']
            # print(begin,text_label,text)
            text_label[begin] = 'B-NAME'   # 在这个地方，在使用lstm进行预测时，建议将相关的名称改成i_name,使用bert模型时，直接使用bio标注方式
            text_label[begin+1:end] = ['I-NAME' for label in range(end-begin-1)]

        text_label = text_label[:len(text)]
        for word_, label_ in zip(text, text_label):
            dataname_out.write(f'{word_} {label_}\n')
        dataname_out.write('\n')


dataname_out.close()


