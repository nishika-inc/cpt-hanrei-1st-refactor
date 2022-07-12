import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


loss_fct = CrossEntropyLoss()


class NERStackingModel(nn.Module):
    def __init__(self, hidden_dim, model_num):
        super().__init__()
        # 4096次元がFlair Embedding
        # 11*model_num次元がBertモデルのLogits。11は今回予測するタグの数（utils.py参照）。今回はmodel_num:モデルの数は6
        self.embedding_lstm = nn.LSTM(
            4096 + 11 * model_num, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(11 * model_num, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 11),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, input_logits, embeddings, labels=None):
        # 6つのBertモデルのLogitsを連結した行列をインプットにBiLSTMにかけた結果と、
        # 6つのBertモデルのLogitsとFlair Embeddingを連結した行列をインプットにBiLSTMにかけた結果を、連結
        concated = torch.cat([input_logits, embeddings], -1)
        embedding_out, _ = self.embedding_lstm(concated)
        logits_out, _ = self.lstm(input_logits)
        features = torch.cat([logits_out, embedding_out], -1)
        # Linear->Relu->Linearにかける
        # 5つの平均を取っているのは色々なdropoutで平均を取りたいため(Multi-Sample Dropout: https://arxiv.org/abs/1905.09788)
        logits = torch.mean(
            torch.stack(
                [self.logits(self.high_dropout(features)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        if labels is None:
            return (logits,)
        # view: 1つ目の引数に-1を入れることで、2つ目の引数で指定した値にサイズ数を自動的に調整
        loss = loss_fct(logits.view(-1, 11), labels.view(-1))
        return loss, logits


def _extract_f1_from_report(report):
    return float(report.split()[report.split().index("micro") + 4])
