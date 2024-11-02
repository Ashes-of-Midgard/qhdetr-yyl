# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        """Input:
            - tensor_list: NestedTensor
                - tensor_list.tensors: Tensor, shape=[batch_size, 3, H, W]
                - tensor_list.mask: Tensor, shape=[batch_size, H, W]
           
           Output:
            - Tensor, shape=[batch_size, d_model, H, W]
        """
        x = tensor_list.tensors
        # x: Tensor, shape=[batch_size, 3, H, W]
        mask = tensor_list.mask
        # mask: Tensor, shape=[batch_size, H, W]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # y_embed: Tensor, shape=[batch_size, H, W]
        # 1    1    1 ...1    0    0
        # 2    2    2 ...2    0    0
        #  ...        ...     ...
        # h    h    h ...h    0    0
        # 0    0    0 ...0    0    0
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # x_embed: Tensor, shape=[batch_size, H, W]
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        
        # self.num_pos_feats = d_model//2
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t=[0, 1, ..., d_model//2 - 1]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # dim_t: Tensor, shape=[d_model//2]

        pos_x = x_embed[:, :, :, None] / dim_t
        # pos_x: Tensor, shape=[batch_size, H, W, d_model//2]
        pos_y = y_embed[:, :, :, None] / dim_t
        # pos_y: Tensor, shape=[batch_size, H, W, d_model//2]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # pos_x: Tensor, shape=[batch_size, H, W, d_model//2]
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # pos_y: Tensor, shape=[batch_size, H, W, d_model//2]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # pos: Tensor, shape=[batch_size, d_model, H, W]
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
