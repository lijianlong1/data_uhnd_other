import json, time
from tqdm import tqdm


class Metrics(object):
    """用于实体级别评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, std_tags, predict_tags):
        """
        初始化对照文件中的标签列表、预测文件的标签列表、以及
        :param std_tags:
        :param predict_tags:
        """
        # 将按句的标签列表转化成按字的标签列表 如 [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.std_tags = flatten_lists(std_tags)  # 将已知标签和预测的标签拼成列表
        self.predict_tags = flatten_lists(predict_tags)
        # self.std_tags = self.bio_to_bmes(std_tags)
        # self.predict_tags = self.bio_to_bmes(predict_tags)
        self.label = set()
        # for label in self.std_tags:
        #     self.label.add(label)
        # for label in self.predict_tags:
        #     self.label.add(label)
        # self.label.(self.std_tags)
        # self.label.add(self.predict_tags)
        # 计数器
        self.std_entity_counter = self.count_entity_dict(self.std_tags)  # 标准结果的各实体个数
        self.predict_entity_counter = self.count_entity_dict(self.predict_tags)  # 预测结果的各实体个数

        print("标准各实体个数", self.std_entity_counter)
        print("预测各实体个数", self.predict_entity_counter)

        self.std_entity_number = self.count_entity(self.std_tags)  # 标准结果的实体总个数
        self.predict_entity_number = self.count_entity(self.predict_tags)  # 预测结果的实体总个数
        print("标准实体数", self.std_entity_number)
        print("预测实体数", self.predict_entity_number)

        self.corrent_entity_number = self.count_correct_entity()
        print("正确的实体", self.corrent_entity_number)

        self.entity_set = set(self.std_entity_counter)
        print("实体集合", self.entity_set)

        # 计算精确率
        self.precision_scores = self.cal_precision()
        print("各个实体的准确率", self.precision_scores)

        # 计算召回率
        self.recall_scores = self.cal_recall()
        print("各个实体的召回率", self.recall_scores)

        # 计算F1分数
        self.f1_scores = self.cal_f1()
        print("各个实体的f1值", self.f1_scores)

        # 计算加权均值
        self.wighted_average = self._cal_wighted_average()
        print("各项指标的加权均值", self.wighted_average)

    def bio_to_bmes(self, tags):
        sign_B = 0
        sign_I = 0
        for i in range(len(tags)):
            label = tags[i]
            if label[0] == 'B':
                if sign_I == 1:
                    label = tags[i - 1]
                    label = list(label)
                    label[0] = 'E'
                    label = ''.join(label)
                    tags[i - 1] = label
                elif sign_B == 1:
                    label = tags[i - 1]
                    label = list(label)
                    label[0] = 'S'
                    label = ''.join(label)
                    tags[i - 1] = label
                if i == len(tags) - 1:
                    tags[i] = 'S'
                sign_B = 1
                sign_I = 0
            elif label[0] == 'I':
                label = list(label)
                label[0] = 'M'
                label = ''.join(label)
                tags[i] = label
                sign_B = 0
                sign_I = 1
                if i == len(tags) - 1:
                    label = list(label)
                    label[0] = 'E'
                    label = ''.join(label)
                    tags[i] = label
            elif label[0] == 'O' and i != 0:
                if sign_B == 1:
                    tags[i - 1] = 'S'
                if sign_I == 1:
                    label = tags[i - 1]
                    label = list(label)
                    label[0] = 'E'
                    label = ''.join(label)
                    tags[i - 1] = label
                sign_I = 0
                sign_B = 0
        return tags

    # def bio_to_bmes2(self, tags):
    #     for i in range(len(tags)):
    #         label=tags[i]
    #         if label[0] == 'B':
    #             if i+1<len(tags) and tags[i+1][0] == 'O':
    #                 label = 'S'+label[1:]
    #             elif i==len(tags)-1:

    # if (label[0] == 'B'):
    #     if sen[index + 1][1][0] == 'I':
    #         label = label
    #     elif sen[index + 1][1][0] == 'O':
    #         label = 'S' + label[1:]
    # elif (label[0] == 'I'):
    #     if sen[index + 1][1][0] == 'I':
    #         label = 'M' + label[1:]
    #     if sen[index + 1][1][0] == 'O' or sen[index + 1][1][0] == 'B':
    #         label = 'E' + label[1:]
    # elif (label[0] == 'O'):
    #     label = label

    def cal_precision(self):
        # 计算每类实体的准确率
        precision_scores = {}
        # 某实体正确的个数  /  预测中某实体所有的个数
        for entity in self.entity_set:
            if entity in self.predict_entity_counter:
                precision_scores[entity] = self.corrent_entity_number.get(entity, 0) / max(1e-10,
                                                                                           self.predict_entity_counter[
                                                                                               entity])
            else:
                precision_scores[entity] = self.corrent_entity_number.get(entity, 0) / 1e-10

        return precision_scores

    def cal_recall(self):
        # 计算每类尸体的召回率
        recall_scores = {}
        for entity in self.entity_set:
            recall_scores[entity] = self.corrent_entity_number.get(entity, 0) / max(1e-10,
                                                                                    self.std_entity_counter[entity])
        return recall_scores

    def cal_f1(self):
        # 计算f1值
        f1_scores = {}
        for entity in self.entity_set:
            p, r = self.precision_scores[entity], self.recall_scores[entity]
            f1_scores[entity] = 2 * p * r / (p + r + 1e-10)
        return f1_scores

    def report_scores(self):
        """
        将结果用表格的形式打印出来
        :return:
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))
        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个实体的p, r, f
        for entity in self.entity_set:
            print(row_format.format(
                entity,
                self.precision_scores[entity],
                self.recall_scores[entity],
                self.f1_scores[entity],
                self.std_entity_counter[entity]  # 这部分是support的值
            ))
        # 计算并打印平均值
        avg_metrics = self._cal_wighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            self.std_entity_number
        ))

    def _cal_wighted_average(self):
        # 计算加权均值

        weighted_average = {}
        total = self.std_entity_number  # 标准实体的总数

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for entity in self.entity_set:
            size = self.std_entity_counter[entity]  # 标准结果各个实体的个数
            weighted_average['precision'] += self.precision_scores[entity] * size
            weighted_average['recall'] += self.recall_scores[entity] * size
            weighted_average['f1_score'] += self.f1_scores[entity] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    # 以下注释掉的代码为将B-company E-book这种BMSE格式符合情况但标签前后不一致，不计入预测结果的实体数中
    # def count_entity_dict(self, tag_list):
    #     """
    #     计算每个实体对应的个数，注意BME、BE、S结构才为实体，其余均不计入,B-company和E-book也不计入
    #     :param tag_list:
    #     :return:
    #     """
    #     enti_dict = {}
    #     flag = 0  # 初始状态设置为0
    #     B_word = ''  #初始状态B设为空
    #     for tag in tag_list:
    #         if 'B-' in tag and flag == 0:  #当B-出现时，将状态变为1
    #             flag = 1
    #             B_word = tag[2:]
    #         if 'M-' in tag and flag == 1:
    #             M_word = tag[2:]
    #             if M_word != B_word:   #当M和B标签不同时，不为实体将B_word设为空
    #                 B_word = ''
    #                 flag = 0
    #         if 'E-' in tag and flag == 1: #E前有B则可以判定为一个实体
    #             flag = 0  #状态回归初始
    #             E_word = tag[2:]
    #             tag = tag[2:]
    #             if E_word == B_word:
    #                 if tag not in enti_dict:
    #                     enti_dict[tag] = 1
    #                 else:
    #                     enti_dict[tag] += 1
    #             B_word = ''
    #         if 'S-' in tag:  #当S-出现，直接加一
    #             B_word = ''
    #             flag = 0
    #             tag = tag[2:]
    #             if tag not in enti_dict:
    #                 enti_dict[tag] = 1
    #             else:
    #                 enti_dict[tag] += 1
    #         if 'O' in tag: #出现O-时，将状态设为0 B_word设为0
    #             B_word = ''
    #             flag = 0
    #     return enti_dict

    def count_entity_dict(self, tag_list):
        """
        计算每个实体对应的个数，注意BME、BE、S结构才为实体，其余均不计入 B-company E-company算实体
        :param tag_list:
        :return:
        """
        enti_dict = {}
        flag = 0  # 初始状态设置为0
        for tag in tag_list:
            if 'B-' in tag and flag == 0:  # 当B-出现时，将状态变为1
                flag = 1
            if 'E-' in tag and flag == 1:  # E前有B则可以判定为一个实体
                flag = 0  # 状态回归初始
                tag = tag[2:]
                if tag not in enti_dict:
                    enti_dict[tag] = 1
                else:
                    enti_dict[tag] += 1
            if 'S-' in tag:  # 当S-出现，直接加一
                flag = 0
                tag = tag[2:]
                if tag not in enti_dict:
                    enti_dict[tag] = 1
                else:
                    enti_dict[tag] += 1
            if tag == 'O':  # 出现O-时，将状态设为0 B_word设为0
                flag = 0
        return enti_dict

    def count_correct_entity(self):
        """
        计算每种实体被正确预测的个数
        address、book、company、game、government、movie、name、organization、position、scene
        :return:
        """
        correct_enti_dict = {}
        flag = 0  # 初始状态，表示等待下一个开始状态
        for std_tag, predict_tag in zip(self.std_tags, self.predict_tags):  # zip()用于将数据打包成元组
            if 'B-' in std_tag and std_tag == predict_tag and flag == 0:  # 当以B-开头且标签相等时
                flag = 1  # 表示已经有B
            if 'M-' in std_tag and std_tag == predict_tag and flag == 1:  # M前已经有过B
                flag = 1
            if 'E-' in std_tag and std_tag == predict_tag and flag == 1:
                flag = 0  # 状态重新调整回初始值
                std_tag = std_tag[2:]
                if std_tag not in correct_enti_dict:
                    correct_enti_dict[std_tag] = 1
                else:
                    correct_enti_dict[std_tag] += 1
            if 'S-' in std_tag and std_tag == predict_tag and flag == 0:
                std_tag = std_tag[2:]
                if std_tag not in correct_enti_dict:
                    correct_enti_dict[std_tag] = 1
                else:
                    correct_enti_dict[std_tag] += 1
            if std_tag != predict_tag:
                flag = 0
        return correct_enti_dict

    def count_entity(self, tag_list):
        """
        计算标准列表中的实体个数，因为标准结果中无错误分类，所以实体的个数可以直接计算为E标签和S标签数目之和
        :return:
        """
        entity_count = 0  # 记录实体数量
        for tag in tag_list:  # 遍历所有标签
            if 'E-' in tag:
                entity_count += 1
            if 'S-' in tag:
                entity_count += 1
        return entity_count


def flatten_lists(lists):
    """
    将列表的列表拼成一个列表
    :param lists:
    :return:
    """
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


def read_file(input_file):
    tag_lists = []
    with open(input_file, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
        for line in tqdm(in_lines):
            line = json.dumps(eval(line))
            label_list = []
            line = json.loads(line)
            text = line['text']
            entity_lists = line['entity_list']
            for i in range(len(text)):
                label_list.append('O')
            for i in range(len(entity_lists)):
                entity_list = entity_lists[i]
                # entity_list = json.loads(entity_list)
                if entity_list == None:
                    continue
                if 'entity_type' not in entity_list:
                    continue
                entity_index = entity_list['entity_index']
                entity_type = entity_list['entity_type']
                begin = int(entity_index['begin'])
                end = int(entity_index['end'])
                if begin >= len(text) or end > len(text):
                    continue
                if end - begin == 1:
                    label_list[begin] = 'S'
                else:
                    label_list[begin] = 'B-' + entity_type
                    begin = begin + 1
                    while begin < end - 1:
                        label_list[begin] = 'M-' + entity_type
                        begin = begin + 1
                    label_list[end - 1] = 'E-' + entity_type
            tag_lists.append(label_list)
    return tag_lists


def write_file(output_file, text, tags_list):
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(text)):
            for j in range(len(text[i])):
                if text[i][j] == '\n':
                    continue
                f.write(f'{text[i][j]} {tags_list[i][j]}\n')
            f.write('\n')


def span_to_bio(input_file):
    tag_lists = []
    text_lists = []
    with open(input_file, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
        for line in tqdm(in_lines):
            line = json.dumps(eval(line))
            label_list = []
            line = json.loads(line)
            text = line['text']
            entity_lists = line['entity_list']
            for i in range(len(text)):
                label_list.append('O')
            for i in range(len(entity_lists)):
                entity_list = entity_lists[i]
                # entity_list = json.loads(entity_list)
                if entity_list == None:
                    continue
                if 'entity_type' not in entity_list:
                    continue
                entity_index = entity_list['entity_index']
                entity_type = entity_list['entity_type']
                begin = int(entity_index['begin'])
                end = int(entity_index['end'])
                if begin >= len(text) or end >= len(text):
                    continue
                # if end - begin == 1:
                #     label_list[begin] = 'S'
                # else:
                label_list[begin] = 'B-' + entity_type
                begin = begin + 1
                while begin < end:
                    label_list[begin] = 'I-' + entity_type
                    begin = begin + 1
                # label_list[end] = 'I-' + entity_type
            tag_lists.append(label_list)
            text_lists.append(text)
    write_file('b_c_m_p_r_w_bio.txt', text_lists, tag_lists)
    return tag_lists


def write_bio_to_bmes(input_file, output_file):
    path = input_file
    res_path = output_file

    f = open(path, encoding='utf-8')
    f1 = open(res_path, 'w+', encoding='utf_8')

    sentences = []
    sentence = []
    label_set = set()
    cnt_line = 0
    for line in f:
        cnt_line += 1
        if len(line) == 0 or line[0] == '\n':
            if len(sentence) > 0:
                sentences.append(sentence)
                print(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0], splits[-1][:-1]])
        label_set.add(splits[-1])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    f.close()

    for sen in sentences:
        i = 0
        for index, word in enumerate(sen):
            char = word[0]
            label = word[1]
            if index < len(sen) - 1:
                if label[0] == 'B':
                    if sen[index + 1][1][0] == 'I':
                        label = label
                    elif sen[index + 1][1][0] == 'O':
                        label = 'S' + label[1:]
                elif label[0] == 'I':
                    if sen[index + 1][1][0] == 'I':
                        label = 'M' + label[1:]
                    if sen[index + 1][1][0] == 'O' or sen[index + 1][1][0] == 'B':
                        label = 'E' + label[1:]
                elif label[0] == 'O':
                    label = label
            else:
                if label[0] == 'B':
                    label = 'S' + label[1:]
                elif label[0] == 'I':
                    label = 'E' + label[1:]
                elif label[0] == 'O':
                    label = label

            f1.write(f'{char} {label}\n')
        f1.write('\n')
    f1.close()


def bio_to_bmes(labels):
    for i in range(len(labels)):
        label = list(labels[i])
        if label[0] == 'I':
            if i != len(labels) - 1 and labels[i + 1][0] == 'I':
                label[0] = 'M'
                label = ''.join(label)
                labels[i] = label
            else:
                label[0] = 'E'
                label = ''.join(label)
                labels[i] = label
        elif label[0] == 'B':
            if i == len(labels) - 1 or labels[i + 1][0] == 'O':
                label[0] = 'S'
                label = ''.join(label)
                labels[i] = label
    return labels


def read_bio_file(input_file):
    label_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
        label_temp = []
        text_temp = []
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 1:
                pairs = line.strip().split()
                word = pairs[0]
                label = pairs[-1]
                label = list(label)
                label = ''.join(label)
                label_temp.append(label)
                text_temp.append(word)
            elif line == '\n':
                label_list.append(bio_to_bmes(label_temp))
                label_temp = []
                text_temp = []
    return label_list[:448]


def read_bmes_file(input_file):
    label_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
        label_temp = []
        text_temp = []
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 1:
                pairs = line.strip().split()
                word = pairs[0]
                label = pairs[-1]
                label = list(label)
                label = ''.join(label)
                label_temp.append(label)
                text_temp.append(word)
            elif line == '\n':
                label_list.append(label_temp)
                label_temp = []
                text_temp = []
    return label_list[:448]


def read_msra_file(input_file):
    label_set = set()
    datalist = []
    label_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
        for idx in range(len(in_lines)):
            label_temp = []
            text_temp = []
            line = in_lines[idx].strip().split()
            if len(line) == 0:
                continue
            for word in line:
                word = word.split('/')
                if word[1] != 'o':
                    if len(word[0]) == 1:
                        text_temp.append(word[0])
                        label_temp.append("B-" + word[1])
                        label_set.add(word[1])
                    else:
                        text_temp.append(word[0][0])
                        label_temp.append("B-" + word[1])
                        label_set.add(word[1])
                        for j in word[0][1:]:
                            text_temp.append(j)
                            label_temp.append("I-" + word[1])
                else:
                    for j in word[0]:
                        text_temp.append(j)
                        label_temp.append("O")
                        label_set.add("O")
                    label_list.append(label_temp)
    return label_list


def f1_score(raw_file, pred_file):
    test_tag_lists = read_file(raw_file)
    pred_tag_lists = read_file(pred_file)
    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores()


def resume_f1_score(raw_file, pred_file):
    pred_label_list = read_bio_file(pred_file)
    # pred_label_list = read_bmes_file(pred_file)
    raw_label_list = read_bmes_file(raw_file)
    metrics = Metrics(raw_label_list, pred_label_list)
    metrics.report_scores()


# resume_f1_score('./dataset/ResumeNER/dev.char.bmes', './BMES.txt')
# resume_f1_score('./dataset/ResumeNER/dev.char.bmes', './resume_pred.txt')
f1_score('msra_test.txt', 'msra_test_out_crf.txt')

# f1_score('boson_1000.txt','boson_1000_gp_fgm_out.txt')
# f1_score('boson_1000.txt','boson_1000_gp_nofgm_out.txt')
