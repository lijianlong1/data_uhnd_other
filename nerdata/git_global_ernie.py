# Coding by long
# Datatime:2022/4/2 21:27
# Filename:global_fgm_train.py
# Toolby: PyCharm
# ______________coding_____________
import warnings
from transformers import get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

import os
import jieba
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from ark_nlp.factory.utils.seed import set_seed
# from ark_nlp.model.ner.global_pointer_bert import GlobalPointerBert
# from ark_nlp.model.ner.global_pointer_bert import EfficientGlobalPointerBert as GlobalPointerBert
from ark_nlp.model.ner.global_pointer_bert import GlobalPointerBertConfig
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Task
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer

from tqdm import tqdm

set_seed(42)
model_path = '../chinese_RoBERT_wwm'

import os
from ark_nlp.factory.utils.conlleval import get_entity_bio

datalist = []
with open('data_out_per_word.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines.append('\n')

    text = []
    labels = []
    label_set = set()

    for line in tqdm(lines):
        if line == '\n':
            text = ''.join(text)
            entity_labels = []
            for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                entity_labels.append({
                    'start_idx': _start_idx,
                    'end_idx': _end_idx,
                    'type': _type,
                    'entity': text[_start_idx: _end_idx + 1]
                })

            if text == '':
                continue

            datalist.append({
                'text': text,
                'label': entity_labels
            })

            text = []
            labels = []

        elif line == '  O\n':
            text.append(' ')
            labels.append('O')
        else:
            line = line.strip('\n').split()
            if len(line) == 1:
                term = ' '
                label = line[0]
            else:
                term, label = line
            text.append(term)
            label_set.add(label.split('-')[-1])
            labels.append(label)

# 这里随意分割了一下看指标，建议实际使用sklearn分割或者交叉验证

train_data_df = pd.DataFrame(datalist[:20000])
train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

dev_data_df = pd.DataFrame(datalist[20000:])
dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

label_list = sorted(list(label_set))

ner_train_dataset = Dataset(train_data_df, categories=label_list)
ner_dev_dataset = Dataset(dev_data_df, categories=ner_train_dataset.categories)

tokenizer = Tokenizer(vocab=model_path, max_seq_len=512)

ner_train_dataset.convert_to_ids(tokenizer)
ner_dev_dataset.convert_to_ids(tokenizer)

config = GlobalPointerBertConfig.from_pretrained(model_path,
                                                 num_labels=len(ner_train_dataset.cat2id))

torch.cuda.empty_cache()

from transformers import BertModel
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer


class EfficientGlobalPointerBert(BertForTokenClassification):
    """
    EfficientGlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8877
        [2] https://github.com/powerycy/Efficient-GlobalPointer
    """  # noqa: ignore flake8"

    def __init__(
            self,
            config,
            encoder_trained=True,
            head_size=64
    ):
        super(EfficientGlobalPointerBert, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.lstm = nn.LSTM(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.efficient_global_pointer = EfficientGlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states

        sequence_output = outputs[-1]
        sequence_output = self.dropout(sequence_output)
        sequence_output, (h_0, c_0) = self.lstm(sequence_output)

        logits = self.efficient_global_pointer(sequence_output, mask=attention_mask)

        return logits


dl_module = EfficientGlobalPointerBert.from_pretrained(model_path,
                                                       config=config)

# 设置运行次数
num_epoches = 4
batch_size = 32  # 32

from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    RMSprop,
    Rprop,
    SGD
)
from torch.optim import Optimizer
from transformers import AdamW


def get_default_bert_optimizer(
        module,
        lr: float = 2e-5,  # 3e-5
        eps: float = 1e-6,
        correct_bias: bool = True,
        weight_decay: float = 1e-3,  # 1e-3
):
    no_decay = ["bias", "LayerNorm.weight"]
    other_params = ["lstm", "classifier", "efficient_global_pointer"]
    bert_params = ["bert"]
    is_main = bert_params + no_decay
    param_group = [
        {'params': [p for n, p in module.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in bert_params)], 'weight_decay': 0,
         'lr': 5e-5},
        {'params': [p for n, p in module.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in bert_params)],
         'weight_decay': weight_decay, 'lr': 5e-5},
        {'params': [p for n, p in module.named_parameters() if not any(nd in n for nd in is_main)],
         'weight_decay': weight_decay, 'lr': 1e-3},
        {'params': [p for n, p in module.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_params)],
         'weight_decay': 0, 'lr': 1e-3},
    ]
    # bert_decay = (
    #     [n for n, p in module.named_parameters() if
    #      any(nd in n for nd in no_decay) and any(nd in n for nd in bert_params)])
    # bert_no_decay = ([n for n, p in module.named_parameters() if
    #                   not any(nd in n for nd in no_decay) and any(nd in n for nd in bert_params)])
    # other_no_decay = ([n for n, p in module.named_parameters() if not any(nd in n for nd in is_main)])
    # other_decay = ([n for n, p in module.named_parameters() if
    #                 any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_params)])
    # all_params = [n for n, p in module.named_parameters()]
    # a = set(bert_decay + bert_no_decay + other_no_decay + other_decay)
    # b = set(all_params)
    # print(a == b)
    # print("a:", a)
    # print("b:", b)
    # bert_params = []
    # lstm_params = []
    # fc_params = []
    # global_params = []
    # other_params = []
    # for name, para in module.named_parameters():
    #     if para.requires_grad:
    #         print("raw:", name)
    #         if "bert" in name:
    #             bert_params += [para]
    #         elif "lstm" in name:
    #             lstm_params += [para]
    #         elif "classifier" in name:
    #             fc_params += [para]
    #         elif "efficient_global_pointer" in name:
    #             global_params += [para]
    #         else:
    #             print("other:", name)
    #             other_params += [para]
    #
    # params = [
    #     # {"params": [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)],
    #     #  "weight_decay": weight_decay},
    #     # {"params": [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)],
    #     #  "weight_decay": 0.0},
    #     {"params": bert_params, "lr": lr},
    #     {"params": lstm_params, "lr": 1e-3},
    #     {"params": fc_params, "lr": 1e-3},
    #     {"params": global_params, "lr": 1e-3},
    #     # {"params": other_params, "lr": lr},
    # ]

    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)],
    #      "weight_decay": weight_decay},
    #     {"params": [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.0},
    # ]
    optimizer = AdamW(param_group,
                      # lr=lr,
                      eps=eps,
                      correct_bias=correct_bias)
    # weight_decay=weight_decay)
    return optimizer


# optimizer = get_default_model_optimizer(dl_module)
optimizer = get_default_bert_optimizer(dl_module)
schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=20000 / batch_size,
                                           num_training_steps=num_epoches * 20000 / batch_size)

from torch.utils.data import DataLoader
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.utils.attack import FGM


class AttackTask(Task):

    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
    ):
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()

        self.module.train()

        self.fgm = FGM(self.module)

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        self.fgm.attack()
        logits = self.module(**inputs)
        _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
        attck_loss.backward()
        self.fgm.restore()

        self._on_backward_record(loss, **kwargs)

        return loss


model = AttackTask(dl_module, optimizer, 'gpce', scheduler=schedule, cuda_device=0)
model.fit(ner_train_dataset,
          ner_dev_dataset,
          # lr=2e-5,  # 2e-5
          epochs=num_epoches,
          batch_size=batch_size
          )

torch.save(model.module.state_dict(), 'global_ernie_att1.pth')

# import json
# import torch
# import numpy as np
#
#
# # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
# # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这
# class GlobalPointerNERPredictor(object):
#     """
#     GlobalPointer命名实体识别的预测器
#
#     Args:
#         module: 深度学习模型
#         tokernizer: 分词器
#         cat2id (:obj:`dict`): 标签映射
#     """  # noqa: ignore flake8"
#
#     def __init__(
#             self,
#             module,
#             tokernizer,
#             cat2id
#     ):
#         self.module = module
#         self.module.task = 'TokenLevel'
#
#         self.cat2id = cat2id
#         self.tokenizer = tokernizer
#         self.device = list(self.module.parameters())[0].device
#
#         self.id2cat = {}
#         for cat_, idx_ in self.cat2id.items():
#             self.id2cat[idx_] = cat_
#
#     def _convert_to_transfomer_ids(
#             self,
#             text
#     ):
#
#         tokens = self.tokenizer.tokenize(text)
#         token_mapping = self.tokenizer.get_token_mapping(text, tokens)
#
#         input_ids = self.tokenizer.sequence_to_ids(tokens)
#         input_ids, input_mask, segment_ids = input_ids
#
#         zero = [0 for i in range(self.tokenizer.max_seq_len)]
#         span_mask = [input_mask for i in range(sum(input_mask))]
#         span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
#         span_mask = np.array(span_mask)
#
#         features = {
#             'input_ids': input_ids,
#             'attention_mask': input_mask,
#             'token_type_ids': segment_ids,
#             'span_mask': span_mask
#         }
#
#         return features, token_mapping
#
#     def _get_input_ids(
#             self,
#             text
#     ):
#         if self.tokenizer.tokenizer_type == 'vanilla':
#             return self._convert_to_vanilla_ids(text)
#         elif self.tokenizer.tokenizer_type == 'transfomer':
#             return self._convert_to_transfomer_ids(text)
#         elif self.tokenizer.tokenizer_type == 'customized':
#             return self._convert_to_customized_ids(text)
#         else:
#             raise ValueError("The tokenizer type does not exist")
#
#     def _get_module_one_sample_inputs(
#             self,
#             features
#     ):
#         return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}
#
#     def predict_one_sample(
#             self,
#             text='',
#             threshold=0
#     ):
#         """
#         单样本预测
#
#         Args:
#             text (:obj:`string`): 输入文本
#             threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
#         """  # noqa: ignore flake8"
#
#         features, token_mapping = self._get_input_ids(text)
#         self.module.eval()
#
#         with torch.no_grad():
#             inputs = self._get_module_one_sample_inputs(features)
#             scores = self.module(**inputs)[0].cpu()
#
#         scores[:, [0, -1]] -= np.inf
#         scores[:, :, [0, -1]] -= np.inf
#
#         entities = []
#
#         for category, start, end in zip(*np.where(scores > threshold)):
#             if end - 1 > token_mapping[-1][-1]:
#                 break
#             if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
#                 entitie_ = {
#                     "start_idx": token_mapping[start - 1][0],
#                     "end_idx": token_mapping[end - 1][-1],
#                     "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
#                     "type": self.id2cat[category]
#                 }
#
#                 if entitie_['entity'] == '':
#                     continue
#
#                 entities.append(entitie_)
#
#         return entities
#

# ner_predictor_instance = GlobalPointerNERPredictor(model.module, tokenizer, ner_train_dataset.cat2id)
#
# predict_results = []
#
# with open('./dataset/test/sample_per_line_preliminary_A.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for _line in tqdm(lines):
#         label = len(_line) * ['O']
#         for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
#             if 'I' in label[_preditc['start_idx']]:
#                 continue
#             if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
#                 continue
#             if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
#                 continue
#
#             label[_preditc['start_idx']] = 'B-' + _preditc['type']
#             label[_preditc['start_idx'] + 1: _preditc['end_idx'] + 1] = (_preditc['end_idx'] - _preditc[
#                 'start_idx']) * [('I-' + _preditc['type'])]
#
#         predict_results.append([_line, label])
#
# with open('global_ernie_att_1.txt', 'w', encoding='utf-8') as f:
#     for _result in predict_results:
#         for word, tag in zip(_result[0], _result[1]):
#             if word == '\n':
#                 continue
#             f.write(f'{word} {tag}\n')
#         f.write('\n')
