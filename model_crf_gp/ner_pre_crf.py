# -- coding: utf-8 --
# @Time : 2022/7/14 23:17
# @Author : long
# @Site : 人工智能研究所
# @File : ner_pre_crf.py
# @Software: PyCharm
import json
import warnings

import torch
from model_crf import CrfBert
from model_crf import BertConfig as CrfBertConfig
from ark_nlp.model.ner.crf_bert import Dataset
from ark_nlp.model.ner.crf_bert import Task
torch.backends.cudnn.enabled = False

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print('hhhhhh')

# ner_dict = {'B-CONT': 0, 'B-EDU': 1, 'B-LOC': 2, 'B-NAME': 3, 'B-ORG': 4, 'B-PRO': 5, 'B-RACE': 6, 'B-TITLE': 7, 'I-CONT': 8, 'I-EDU': 9, 'I-LOC': 10, 'I-NAME': 11, 'I-ORG': 12, 'I-PRO': 13, 'I-RACE': 14, 'I-TITLE': 15, 'O': 16}
def my_method(file_in, model_file, file_out):
    ner_dict = {'B-DATE': 0, 'B-GPE': 1, 'B-ORG': 2, 'B-PERSON': 3, 'I-DATE': 4, 'I-GPE': 5, 'I-ORG': 6, 'I-PERSON': 7, 'O': 8}
    print(ner_dict)
    config = CrfBertConfig.from_pretrained('/opt/lijianlong/gaiic/chinese-roberta-wwm',
                                      num_labels=len(ner_dict))
    model_ner = CrfBert.from_pretrained('/opt/lijianlong/gaiic/chinese-roberta-wwm',
                                        config=config)

    model_ner.to(device=device)
    model_ner.load_state_dict(torch.load(model_file, map_location=device),strict=True)
    # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
    # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这



    from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
    optimizer = get_default_model_optimizer(model_ner)

    model = Task(model_ner, optimizer, 'gpce', cuda_device=3)

    from ark_nlp.model.ner.crf_bert import CrfBertNERPredictor


    from ark_nlp.model.ner.crf_bert import Tokenizer
    tokenizer = Tokenizer(vocab='/opt/lijianlong/gaiic/chinese-roberta-wwm', max_seq_len=256)
    crf_ = CrfBertNERPredictor(model.module, tokenizer, ner_dict)

    # 直接在这里开始测试然后得到相应的数据结果
    from tqdm import tqdm
    predict_results = []
    out_file = open(file_out, 'w', encoding='utf-8')

    with open(file_in, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            _line = eval(_line)
            _line = _line['text'].strip()
            dict_result = {}
            dict_result['text'] = _line
            dict_result['entity_list'] = []
            for _preditc in crf_.predict_one_sample(_line):
                pre_one = {}
                begin = _preditc['start_idx']
                end = _preditc['end_idx']
                entity_type = _preditc['type']
                entity = _preditc['entity']
                # 至此所有的相关的数据都拿到了，后面需要进行更加细致的存储处理
                pre_one['entity_index'] = {'begin': begin, 'end': end + 1}
                pre_one['entity_type'] = entity_type
                pre_one['entity'] = entity
                dict_result['entity_list'].append(pre_one)
            #print(dict_result)
            out_file.write(str(dict_result) + '\n')

        # out_file.write('\n')
    out_file.close()


my_method(file_in='/opt/lijianlong/gaiic/data_sets/people_daily/people_daily_ner_1000.txt', model_file='robert_crf_people.pth', file_out='/opt/lijianlong/gaiic/data_sets/people_daily/people_daily_ner_1000_crf.txt')