# Coding by long
# Datatime:2022/4/2 21:27
# Filename:global_fgm_train.py
# Toolby: PyCharm
# ______________coding_____________
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from ark_nlp.factory.utils.seed import set_seed
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Task
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer

set_seed(42)

from ark_nlp.factory.utils.conlleval import get_entity_bio

torch.backends.cudnn.enabled = False


datalist = []
with open('./data_sets/people_daily/people_daily_ner_train_18357_word_lable.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines.append('\n')

    text = []
    labels = []
    label_set = set()

    for line in lines:
        try:
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
                    # print(line)
                    term, label = line
                # print(text)
                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)
        except Exception as e:
            print(e)


train_data_df = pd.DataFrame(datalist[:18357])


train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

dev_data_df = pd.DataFrame(datalist[18357:])
dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))


label_list = sorted(list(label_set))
print(label_list)

ner_train_dataset = Dataset(train_data_df, categories=label_list)
ner_dev_dataset = Dataset(dev_data_df, categories=ner_train_dataset.categories)


tokenizer = Tokenizer(vocab='/opt/lijianlong/gaiic/chinese-roberta-wwm', max_seq_len=256)


ner_train_dataset.convert_to_ids(tokenizer)
ner_dev_dataset.convert_to_ids(tokenizer)

print(ner_train_dataset.cat2id)
#config = NeZhaConfig.from_pretrained('../torch_bertner/nezha-cn-base',
                                                 #num_labels=len(ner_train_dataset.cat2id))


torch.cuda.empty_cache()


from model import globalpointer_NER
dl_module = globalpointer_NER('/opt/lijianlong/gaiic/chinese-roberta-wwm', n_head=len(label_list))


# 设置运行次数
num_epoches = 5
batch_size = 64



from transformers import AdamW,get_cosine_schedule_with_warmup


def get_default_nezha_optimizer(
        module,
        lr: float = 2e-5,  # 3e-5
        eps: float = 1e-6,
        correct_bias: bool = True,
        weight_decay: float = 1e-3,  # 1e-3
):
    no_decay = ["bias", "LayerNorm.weight"]
    other_params = ['lstm', "classifier", "global_pointer"]
    nezha_params = ["bert"]
    is_main = nezha_params + no_decay
    param_group = [
        {'params': [p for n, p in module.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in nezha_params)], 'weight_decay': 0,
         'lr': 5e-5},
        {'params': [p for n, p in module.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in nezha_params)],
         'weight_decay': weight_decay, 'lr': 5e-5},
        {'params': [p for n, p in module.named_parameters() if not any(nd in n for nd in is_main)],
         'weight_decay': weight_decay, 'lr': 1e-3},
        {'params': [p for n, p in module.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in nezha_params)],
         'weight_decay': 0, 'lr': 1e-3},
    ]
    optimizer = AdamW(param_group,
                      # lr=lr,
                      eps=eps,
                      correct_bias=correct_bias)
    # weight_decay=weight_decay)
    return optimizer


# optimizer = get_default_model_optimizer(dl_module)

optimizer = get_default_nezha_optimizer(dl_module)
schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= len(datalist)/batch_size,
                                           num_training_steps=num_epoches * len(datalist) / batch_size)

from torch.utils.data import DataLoader
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.utils.attack import FGM

import torch.nn.functional as F

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
        # self.pgd = PGD(self.module)  # 尝试加入相应的不一样的扰动函数，进行对抗训练测试

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
        #在此使用pgd方式进行对抗学习扰动

        self.fgm.attack(epsilon=1.0)    # 在这个地方可以调整扰动的数值概率
        logits = self.module(**inputs)
        _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
        attck_loss.backward()
        self.fgm.restore()

        self._on_backward_record(loss, **kwargs)

        return loss

#### 在这个地方尝试直接加入rdrop进行模型的训练，看看效果能否有提升


model = AttackTask(dl_module, optimizer, 'gpce', scheduler=schedule, cuda_device=3)





model.fit(ner_train_dataset,
          ner_dev_dataset,
          # lr=2e-5, 此时已经不需要这个学习率了
          epochs=num_epoches,
          batch_size=batch_size
         )



torch.save(model.module.state_dict(), 'bert_global_fgm_people.pth')


