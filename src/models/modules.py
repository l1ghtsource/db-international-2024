import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TripletMarginWithDistanceLoss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        '''
        Initialize the InfoNCE loss.

        Args:
            temperature (float): Temperature parameter to scale the similarity.
        "'''""
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2, labels):
        '''
        Forward pass for InfoNCE loss.

        Args:
            embeddings1 (torch.Tensor): Embedding tensor for the first set of examples.
            embeddings2 (torch.Tensor): Embedding tensor for the second set of examples.
            labels (torch.Tensor): Labels indicating if pairs are similar or dissimilar.

        Returns:
            torch.Tensor: The computed InfoNCE loss.
        '''
        # normalize the embeddings to unit vectors
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # compute similarity matrix between embeddings
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # extract the positives (diagonal elements in similarity matrix)
        positives = torch.diag(similarity_matrix)

        # create a mask for the negative pairs (all non-diagonal elements)
        neg_mask = ~torch.eye(similarity_matrix.shape[0], dtype=bool, device=similarity_matrix.device)
        negatives = similarity_matrix[neg_mask].view(similarity_matrix.shape[0], -1)

        # concatenate positive and negative similarities for cross entropy loss
        logits = torch.cat([positives.unsqueeze(-1), negatives], dim=1)

        # define the labels for the contrastive loss (positive pairs = 0, negatives = 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)  # compute the cross-entropy loss


# triplet loss class for triplet-based learning
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        '''
        Initialize the Triplet Loss.

        Args:
            margin (float): Margin to enforce distance between positive and negative pairs.
        '''
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        '''
        Forward pass for Triplet Loss.

        Args:
            anchor (torch.Tensor): Embedding for the anchor example.
            positive (torch.Tensor): Embedding for the positive example.
            negative (torch.Tensor): Embedding for the negative example.

        Returns:
            torch.Tensor: The computed triplet loss.
        '''
        triplet_loss = TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),  # cosine similarity as distance metric
            margin=self.margin  # margin to enforce separation
        )

        return triplet_loss(anchor, positive, negative)


# combined loss class combining infonce and triplet Loss
class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.07, triplet_margin=0.3, weights=(0.5, 0.5)):
        '''
        Initialize the Combined Loss (InfoNCE + Triplet Loss).

        Args:
            temperature (float): Temperature for InfoNCE loss.
            triplet_margin (float): Margin for Triplet loss.
            weights (tuple): Weighting for each component loss.
        '''
        super().__init__()
        self.infonce = InfoNCELoss(temperature)
        self.triplet = TripletLoss(triplet_margin)
        self.weights = weights

    def forward(self, embeddings1, embeddings2, labels):
        '''
        Forward pass for combined InfoNCE and Triplet loss.

        Args:
            embeddings1 (torch.Tensor): First set of embeddings.
            embeddings2 (torch.Tensor): Second set of embeddings.
            labels (torch.Tensor): Labels indicating similar or dissimilar pairs.

        Returns:
            torch.Tensor: The combined loss.
        '''
        # compute infonce loss
        infonce_loss = self.infonce(embeddings1, embeddings2, labels)

        # compute triplet loss for positive and negative pairs
        positive_pairs = labels == 1  # positive pairs
        negative_pairs = labels == 0  # negative pairs

        if positive_pairs.sum() > 0 and negative_pairs.sum() > 0:
            # apply triplet loss only for valid positive-negative pairs
            triplet_loss = self.triplet(
                embeddings1[positive_pairs],  # anchors
                embeddings2[positive_pairs],  # positives
                embeddings2[negative_pairs[:positive_pairs.sum()]]  # negatives
            )
        else:
            # if no valid pairs, set triplet loss to 0
            triplet_loss = torch.tensor(0.0, device=embeddings1.device)

        # combine both losses using the specified weights
        return self.weights[0] * infonce_loss + self.weights[1] * triplet_loss
