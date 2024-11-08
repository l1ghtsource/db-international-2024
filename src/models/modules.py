import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2, labels):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        positives = torch.diag(similarity_matrix)

        neg_mask = ~torch.eye(similarity_matrix.shape[0], dtype=bool, device=similarity_matrix.device)
        negatives = similarity_matrix[neg_mask].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives.unsqueeze(-1), negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.07, triplet_margin=0.3, weights=(1.0, 1.0)):
        super().__init__()
        self.infonce = InfoNCELoss(temperature)
        self.triplet = TripletLoss(triplet_margin)
        self.weights = weights

    def forward(self, embeddings1, embeddings2, labels):
        infonce_loss = self.infonce(embeddings1, embeddings2, labels)

        positive_pairs = labels == 1
        negative_pairs = labels == 0

        if positive_pairs.sum() > 0 and negative_pairs.sum() > 0:
            triplet_loss = self.triplet(
                embeddings1[positive_pairs],
                embeddings2[positive_pairs],
                embeddings2[negative_pairs[:positive_pairs.sum()]]
            )
        else:
            triplet_loss = torch.tensor(0.0, device=embeddings1.device)

        return self.weights[0] * infonce_loss + self.weights[1] * triplet_loss
