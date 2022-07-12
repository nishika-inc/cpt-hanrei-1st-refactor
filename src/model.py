import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertConfig, \
        get_linear_schedule_with_warmup,BertForTokenClassification
from torch.optim import AdamW
HIDDEN_SIZE = 768
loss_fct = CrossEntropyLoss()


class NERModel(BertPreTrainedModel):
    def __init__(self, conf):
        super().__init__(conf)
        self.num_labels = conf.num_labels
        self.bert = BertModel(config=conf)

        self.high_dropout = torch.nn.Dropout(p=0.5)
        self.classifier = torch.nn.Linear(HIDDEN_SIZE*2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None,attention_mask=None,labels=None, mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # out = output[1]
        hidden_layers = outputs[2]
        out = torch.cat([hidden_layers[-2], hidden_layers[-1]], dim=-1)
        # # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)
        if labels is None:
            return (logits,)
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

def get_optimizer(model, learning_rate, weight_decay):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_parameters,lr=learning_rate,amsgrad=True )
    return optimizer


