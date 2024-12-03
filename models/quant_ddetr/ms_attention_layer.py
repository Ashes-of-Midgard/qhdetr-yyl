from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.nn import MultiheadAttention
import pdb

from typing import Optional, Tuple, List
from .lsq_plus import *
from ._quan_base_plus import *
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
from torch.overrides import has_torch_function, handle_torch_function
# from ..ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )
    return (n & (n - 1) == 0) and n != 0


def ms_deform_attn_core_pytorch(
    value, v_act : nn.Module, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Parameters:
        - value: (N, Len_in, n_heads, d_model // n_heads)
        - v_act: ActLSQ(nbits_a=4, in_features=n_heads)
        - value_spatial_shapes: (n_levels, 2)
        - sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
        - attention_weights: (N, Len_q, n_heads, n_levels * n_points)
    Return:
        - (N, Len_q, d_model)
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape # [2, 14288, 8, 32]
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape # [2, 14288, 8, 4, 4, 2]
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # value_list: List, len(value_list) = n_levels
    # value_list[i]: (N, H_i*W_i, n_heads, d_model // n_heads)
    # sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
    sampling_grids = 2 * sampling_locations - 1
    # sampling_grids: (N, Len_q, n_heads, n_levels, n_points, 2)
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # (N, H*W, n_heads, d_model//n_heads) -> (N, H*W, d_model) -> (N, d_model, H*W) -> (N*n_heads, d_model//n_heads, H, W)
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        )
        # value_l: (N*n_heads, d_model//n_heads, H, W)
        # (N, Len_q, n_heads, n_points, 2) -> (N, n_heads, Len_q, n_points, 2) -> (N*n_heads, Len_q, n_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # sampling_grid_l_: (N*n_heads, Len_q, n_points, 2)
        # (N*n_heads, d_model//n_heads, Len_q, n_points)
        sampling_value_l_ = F.grid_sample(
            value_l_, # (N*n_heads, d_model//n_heads, H, W)
            sampling_grid_l_, # (N*n_heads, Len_q, n_points, 2)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N, Len_q, n_heads, n_levels, n_points) -> (N, n_heads, Len_q, n_levels, n_points) -> (N, n_heads, 1, Len_q, n_levels*n_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    # attention_weights: (N*n_heads, 1, Len_q, n_levels*n_points)
    # sampling_value_list  [4 16 32 14288 4]
    value_mult = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    # value_mult: (N*n_heads, d_model//n_heads, Len_q, n_levels*n_points)
    value_mult = v_act(value_mult)
    output = (
        (value_mult * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    # (N*n_heads, d_model//n_heads, Len_q, n_levels*n_points) * (N*n_heads, 1, Len_q, n_levels*n_points) -> (N*n_heads, d_model//n_heads, Len_q, n_levels*n_points)
    # -> (N, d_model, Len_q)
    # output = (
    #     (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
    #     .sum(-1)
    #     .view(N_, M_ * D_, Lq_)
    # )
    return output.transpose(1, 2).contiguous() # (N, Len_q, d_model)



class QuantMS_DAttention(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, n_bit=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(
                    d_model, n_heads
                )
            )
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # self.q_act = ActLSQ(nbits_a=4, in_features=n_heads)
        # self.k_act = ActLSQ(nbits_a=4, in_features=n_heads)
        self.v_act = ActLSQ(nbits_a=4, in_features=n_heads)
        self.attn_act = ActLSQ(nbits_a=4, in_features=n_heads)

        self.sampling_offsets = LinearLSQ(d_model, n_heads * n_levels * n_points * 2, nbits_w = n_bit, bias=True)
        self.attention_weights = LinearLSQ(d_model, n_heads * n_levels * n_points, nbits_w = n_bit, bias=True)
        self.value_proj = LinearLSQ(d_model, d_model, nbits_w = n_bit, bias=True)
        self.output_proj = LinearLSQ(d_model, d_model, nbits_w = n_bit, bias=True)
        # #################
        # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # self.value_proj = nn.Linear(d_model, d_model)
        # self.output_proj = nn.Linear(d_model, d_model)
        # pdb.set_trace()
        self._reset_parameters()

    def _reset_parameters(self):
        # pdb.set_trace()
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        :param query                       (N, Length_{query}, C)                   4个flatten后的特征图+4个flatten后特征图对应的位置编码 = src_flatten + lvl_pos_embed_flatten
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)    4个flatten后的特征图
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # input_flatten: (N, Len_in, d_model)
        value = self.value_proj(input_flatten) # self.value_proj: LinearLSQ(d_model, d_model, nbits_w = n_bit, bias=True)
        # value: (N, Len_in, d_model)
        if input_padding_mask is not None:
            # input_padding_mask: (N, Len_in)
            value = value.masked_fill(input_padding_mask[..., None], float(0))
            # value: (N, Len_in, d_model), for those elements in padded area, features' value are set to 0 by force.(They should be 0 in ideal assumption)
        
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # value: (N, Len_in, n_heads, d_model // n_heads)
        # query: (N, Len_q, d_model)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        ) # self.sampling_offsets: LinearLSQ(d_model, n_heads * n_levels * n_points * 2, nbits_w = n_bit, bias=True)
        # sampling_offsets: (N, Len_q, n_heads, n_levels, n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        ) # self.attention_weights: LinearLSQ(d_model, n_heads * n_levels * n_points, nbits_w = n_bit, bias=True)
        # attention_weights: (N, Len_q, n_heads, n_levels * n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:          # one-stage
            # input_spatial_shapes: (n_levels, 2)
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            # offset_normalizer: (n_levels, 2), switch two columns
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
            # (N, Len_q, 1, n_levels, 1, 2) + (N, Len_q, n_heads, n_levels, n_points, 2) / (1, 1, 1, n_levels, 1, 2) -> (N, Len_q, n_heads, n_levels, n_points, 2)
            # sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
        elif reference_points.shape[-1] == 4:     # two stage  +  iterative bounding box refinement
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        # output = MSDeformAttnFunction.apply(
        #     value,
        #     input_spatial_shapes,
        #     input_level_start_index,
        #     sampling_locations,
        #     attention_weights,
        #     self.im2col_step,
        # )

        # sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
        # attention_weights[2, 14288, 8, 4, 4]  value[2, 14288, 8, 32]
        # attention_weights: (N, Len_q, n_heads, n_levels * n_points)
        attention_weights = self.attn_act(attention_weights)
        # attention_weights: (N, Len_q, n_heads, n_levels * n_points), quantized
        # value = self.v_act(value)
        output = ms_deform_attn_core_pytorch(
            value, # (N, Len_in, n_heads, d_model // n_heads)
            self.v_act, # ActLSQ(nbits_a=4, in_features=n_heads)
            input_spatial_shapes, # (n_levels, 2)
            sampling_locations, # (N, Len_q, n_heads, n_levels, n_points, 2)
            attention_weights, # (N, Len_q, n_heads, n_levels * n_points)
        )
        output = self.output_proj(output) # self.output_proj: LinearLSQ(d_model, d_model, nbits_w = n_bit, bias=True)
        # output: (N, Len_q, d_model)
        return output
 