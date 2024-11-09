import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
import os
from torchvision import transforms
from torch.nn import TripletMarginWithDistanceLoss
import torch.nn.functional as F
from data.dataloader import get_rkn_dataloader
from models.modules import CombinedLoss
from models.cross_attn_model import CombinedModel
from transformers import CLIPModel


def train_process(config):
    '''
    Function for training model w/ custom loss and parameters.

    Args:
        config (dict): Configuration dictionary with training settings.
    '''
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),              
        transforms.RandomHorizontalFlip(p=0.5),      
        transforms.ColorJitter(brightness=0.2, contrast=0.2),        
        transforms.ToTensor(),                      
    ])

    train_loader, val_loader = get_rkn_dataloader(
        root_dir=config['data']['base_path'],
        transform=resize_transform,
        shuffle=config['dataloader']['shuffle']
    )
    
    if config['training']['model'] == 'clip':
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    elif config['training']['model'] == 'crossattn':
        model = CombinedModel(
            clip_model_name=config['model']['clip_model_name'],
            blip_model_name=config['model']['blip_model_name'],
            num_heads=config['model']['num_heads']
        )

    device = config['training']['device']
    model.to(device)

    lr = float(config['training']['learning_rate'])
    num_epochs = config['training']['num_epochs']
    margin = float(config['loss']['margin'])
    temperature = float(config['loss']['temperature'])
    save_model_path = config['training']['save_model_path']
    early_stopping_patience = config['training']['early_stopping_patience']
    
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)
    
    if config['training']['loss'] == 'triplet':
        loss_func = TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=margin
        )
    elif config['training']['loss'] == 'combined':
        loss_func = CombinedLoss(temperature=temperature, triplet_margin=margin)

    wandb.init(project=config['wandb']['project'], config={
        'learning_rate': lr,
        'epochs': num_epochs,
        'batch_size': train_loader.batch_size,
        'margin': margin,
    })

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for i, (anchor, positive, negative) in enumerate(train_loader):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_embeddings = model.get_image_features(pixel_values=anchor)
                positive_embeddings = model.get_image_features(pixel_values=positive)
                negative_embeddings = model.get_image_features(pixel_values=negative)

                loss = loss_func(anchor_embeddings, positive_embeddings, negative_embeddings)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({'train_loss': loss.item()})
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss/len(train_loader)}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_embeddings = model.get_image_features(pixel_values=anchor)
                positive_embeddings = model.get_image_features(pixel_values=positive)
                negative_embeddings = model.get_image_features(pixel_values=negative)

                loss = loss_func(anchor_embeddings, positive_embeddings, negative_embeddings)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss}')
            wandb.log({'val_loss': val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_model_path)
                print(f'Best model saved with validation loss: {best_val_loss}')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f'No improvement for {epochs_without_improvement} epochs')

            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    wandb.finish()