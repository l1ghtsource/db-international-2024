import torch
import torch.nn as nn
from modules import CrossAttention


class CombinedModel(nn.Module):
    def __init__(self, clip_emb_size=512, blip_emb_size=768, num_heads=8):
        super().__init__()

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

    def forward(self, clip_emb, blip_emb):
        blip_proj = self.blip_proj(blip_emb)
        clip_proj = self.clip_proj(clip_emb)

        clip_to_blip_attn = self.clip_to_blip_cross_attn(
            x=clip_emb,
            k=blip_proj,
            v=blip_proj
        )  # (B, N_clip, C_clip)

        blip_to_clip_attn = self.blip_to_clip_cross_attn(
            x=blip_emb,
            k=clip_proj,
            v=clip_proj
        )  # (B, N_blip, C_blip)

        # avg pooling
        clip_attn_pooled = clip_to_blip_attn.mean(dim=1)  # (B, C_clip)
        blip_attn_pooled = blip_to_clip_attn.mean(dim=1)  # (B, C_blip)

        # concat pooled embeddings
        cat_embs = torch.cat([
            clip_attn_pooled,
            blip_attn_pooled
        ], dim=-1)  # (B, C_clip + C_blip)

        # project to final embedding size
        combined_emb = self.out_proj(cat_embs)  # (B, C_blip)

        return combined_emb
