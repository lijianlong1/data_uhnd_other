# -*- ecoding: utf-8 -*-
# @ModuleName: data_precess
# @Function: 
# @Author: long
# @Time: 2022/9/21 14:27
# *****************conding***************
def data_pre_for_train(file_input, file_out):
    data_out = open(file_out, 'w', encoding='utf-8')
    count = 1
    with open(file_input, 'r', encoding='utf-8') as f:
        data_line = f.readlines()
        for i in data_line:
            i_dict = eval(i)
            sentence = i_dict['text']
            if sentence != '':
                label = ['O'] * len(sentence)
                # if i_dict['entity_list']==[]:
                #     continue
                # else:
                for i in i_dict['entity_list']:
                    start = i['entity_index']['begin']
                    end = i['entity_index']['end']
                    label[start] = 'B-' + i['entity_type']
                    label[start + 1:end] = ['I-' + i['entity_type']] * (end - start - 1)
                for i, j in zip(sentence, label):
                    data_out.write(str(i) + ' ' + j + '\n')

                data_out.write('\n')
                count += 1
        data_out.close()
    print(count)
# 10749
# 1344

data_pre_for_train('train.txt', 'train_word.txt')
data_pre_for_train('dev.txt', 'dev_word.txt')