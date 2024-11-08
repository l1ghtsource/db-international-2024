import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def visualize_random_triplets(dataloader: DataLoader, num_triplets: int = 5):
    '''
    Visualizes random triplets (anchor, positive, negative) from the DataLoader.

    Args:
        dataloader (DataLoader): DataLoader object containing the triplets.
        num_triplets (int): Number of random triplets to visualize.

    Example:
        visualize_random_triplets(train_loader, num_triplets=5)
    '''
    # ensure the figure is properly sized
    fig, axes = plt.subplots(num_triplets, 3, figsize=(12, num_triplets * 4))

    for i in range(num_triplets):
        # get a random triplet (anchor, positive, negative) from the dataloader
        anchor, positive, negative = next(iter(dataloader))

        # choose a random sample from the batch
        idx = random.randint(0, len(anchor) - 1)

        # plot anchor image
        axes[i, 0].imshow(anchor[idx].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Anchor')
        axes[i, 0].axis('off')

        # plot positive image
        axes[i, 1].imshow(positive[idx].permute(1, 2, 0).numpy())
        axes[i, 1].set_title('Positive')
        axes[i, 1].axis('off')

        # plot negative image
        axes[i, 2].imshow(negative[idx].permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Negative')
        axes[i, 2].axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()
