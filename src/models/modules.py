import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossWithHardMining(nn.Module):
    '''
    Triplet loss с hard mining. Ищем самые сложные триплеты, где:
    - Положительный пример ближе к якорю, чем отрицательный пример.
    - Выбираем отрицательный пример, который наиболее близок к якорю.
    '''

    def __init__(self, margin: float = 0.2):
        super(TripletLossWithHardMining, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
