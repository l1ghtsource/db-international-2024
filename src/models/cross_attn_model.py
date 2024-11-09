import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, BlipVisionModel
from models.modules import CrossAttention


class CombinedModel(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        blip_model_name="Salesforce/blip-image-captioning-base",
        num_heads=8
    ):
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.blip_vision_model = BlipVisionModel.from_pretrained(blip_model_name)

        clip_emb_size = self.clip_model.config.vision_config.hidden_size  # typically 512
        blip_emb_size = self.blip_vision_model.config.hidden_size  # typically 768

        # cross attn clip->blip
        self.clip_to_blip_cross_attn = CrossAttention(
            dim=clip_emb_size,
            num_heads=num_heads,
            out_dim=clip_emb_size
        )

        # cross attn blip->clip
        self.blip_to_clip_cross_attn = CrossAttention(
            dim=blip_emb_size,
            num_heads=num_heads,
            out_dim=blip_emb_size
        )

        self.blip_proj = nn.Linear(blip_emb_size, clip_emb_size)
        self.clip_proj = nn.Linear(clip_emb_size, blip_emb_size)

        total_emb_size = clip_emb_size + blip_emb_size
        self.out_proj = nn.Linear(total_emb_size, blip_emb_size)

    def forward(self, images):
        '''
        Args:
            images: Tensor of shape (batch_size, channels, height, width)
        Returns:
            combined_emb: Tensor of shape (batch_size, 768)
        '''
        clip_features = self.clip_model.get_image_features(images)  # (B, 512)
        clip_emb = clip_features.unsqueeze(1)  # (B, 1, 512)

        blip_outputs = self.blip_vision_model(images)
        blip_emb = blip_outputs.last_hidden_state  # (B, num_patches, 768)

        blip_proj = self.blip_proj(blip_emb)
        clip_proj = self.clip_proj(clip_emb)

        clip_to_blip_attn = self.clip_to_blip_cross_attn(
            x=clip_emb,
            k=blip_proj,
            v=blip_proj
        )  # (B, 1, 512)

        blip_to_clip_attn = self.blip_to_clip_cross_attn(
            x=blip_emb,
            k=clip_proj,
            v=clip_proj
        )  # (B, N_patches, 768)

        clip_attn_pooled = clip_to_blip_attn.squeeze(1)  # (B, 512)
        blip_attn_pooled = blip_to_clip_attn.mean(dim=1)  # (B, 768)

        cat_embs = torch.cat([
            clip_attn_pooled,
            blip_attn_pooled
        ], dim=-1)  # (B, 1280)

        combined_emb = self.out_proj(cat_embs)  # (B, 768)

        return combined_emb
