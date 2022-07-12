import torch
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
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



