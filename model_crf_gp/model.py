# Coding by long
# Datatime:2022/4/8 20:10
# Filename:model.py
# Toolby: PyCharm
# description:
# ______________coding_____________
import torch.nn as nn
from transformers import BertModel, BertConfig
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer


class globalpointer_NER(nn.Module):
    def __init__(self, nezha_path, n_head):
        super(globalpointer_NER, self).__init__()
        self.config = BertConfig.from_pretrained(nezha_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=nezha_path,config=self.config)  # 加载预训练模型权重.使用albert轻量级模型进行实体识别训练预测
        self.lstm = nn.GRU(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()  # 使用相应的激活函数
        # self.softmax = nn.Softmax(-1) # 表示在最后一个维度进行变化

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state
       # print((outputs.shape))
        sequence_output = outputs
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        #sequence_output, (_, _) = self.lstm(sequence_output)
        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits   # shape(batch-size, 512, 4) 每次训练的时候，考虑到消耗的问题，每次训练的batch为1或者2就可以了，不然会溢出