import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from torchvision import transforms
from data.dataloader import get_rkn_dataloader
from models.modules import TripletLossWithHardMining
from transformers import CLIPModel


def train_clip_with_triplet_loss(config):
    '''
    Обучение CLIP модели с использованием triplet loss с hard mining.

    Args:
        train_loader (DataLoader): DataLoader для обучающих данных.
        val_loader (DataLoader): DataLoader для валидационных данных.
        model (CLIPModel): CLIP модель.
        processor (CLIPProcessor): CLIP процессор.
        device (torch.device): Устройство для обучения (например, 'cuda' или 'cpu').
        num_epochs (int): Количество эпох для обучения.
        lr (float): Скорость обучения.
        margin (float): Margin для triplet loss.
        save_model_path (str): Путь для сохранения обученной модели.
    '''

    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_loader, val_loader = get_rkn_dataloader(root_dir=config['data']['base_path'],
                                                  transform=resize_transform,
                                                  shuffle=True)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    device = config['training']['device']
    model.to(device)

    lr = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    margin = config['training']['margin']
    save_model_path = config['training']['save_model_path']

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)
    triplet_loss = TripletLossWithHardMining(margin=margin).to(device)

    wandb.init(project="clip-triplet-training", config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "margin": margin,
    })

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i, (anchor, positive, negative) in enumerate(train_loader):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_embeddings = model.get_image_features(pixel_values=anchor)
                positive_embeddings = model.get_image_features(pixel_values=positive)
                negative_embeddings = model.get_image_features(pixel_values=negative)

                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({"train_loss": loss.item()})
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss/len(train_loader)}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_embeddings = model.get_image_features(pixel_values=anchor)
                positive_embeddings = model.get_image_features(pixel_values=positive)
                negative_embeddings = model.get_image_features(pixel_values=negative)

                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss}")
            wandb.log({"val_loss": val_loss})

        torch.save(model.state_dict(), save_model_path)

    wandb.finish()
