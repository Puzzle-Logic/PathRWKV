import torch
from torch import nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, device="cpu"):
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0**omega)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device="cpu"):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], device=device
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], device=device
    )  # (H*W, D/2)
    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_token=0, device="cpu"):
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)

    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)

    grid = grid.reshape(2, -1)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device=device)

    if add_token > 0:
        token_embed = torch.zeros(add_token, embed_dim, device=device)
        pos_embed = torch.cat([token_embed, pos_embed], dim=0)

    return pos_embed


class SlidePE(nn.Module):
    def __init__(self, embed_dim, slide_ngrids, MTL_token_num=0):
        super().__init__()
        self.slide_ngrids = slide_ngrids
        self.MTL_token_num = MTL_token_num

        total_pos_len = MTL_token_num + slide_ngrids**2
        self.register_buffer(
            "pos_embed", torch.zeros(1, total_pos_len, embed_dim), persistent=False
        )
        pos_embed_weights = get_2d_sincos_pos_embed(
            embed_dim,
            slide_ngrids,
            add_token=MTL_token_num,
            device=self.pos_embed.device,
        )
        self.pos_embed.data.copy_(pos_embed_weights.float().unsqueeze(0))

    def coords_to_pos(self, coords):
        coords = torch.floor(coords / 224)
        pos = coords[..., 0] * self.slide_ngrids + coords[..., 1]
        return pos.long() + self.MTL_token_num

    def forward(self, x, coords, MTL_tokens=None):
        pos_ids = self.coords_to_pos(coords)
        x = x + self.pos_embed[:, pos_ids, :].squeeze(0)
        if MTL_tokens is not None:
            MTL_tokens = (
                MTL_tokens.expand(x.shape[0], -1, -1)
                + self.pos_embed[:, : self.MTL_token_num, :]
            )
            x = torch.cat((MTL_tokens, x), dim=1)  # [B, T+N, D]
        return x
