# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build
from .quant_ddetr.quant_ddetr import build_quant
from .quant_ddetr.quant_ddetr_dis import build_quant_dis
from .quant_ddetr.quant_ddetr_merge import build_quant_merge


def build_model(args):
    return build(args)

def build_quant_model(args):
    return build_quant(args)

def build_quant_model_dis(args):
    return build_quant_dis(args)

def build_quant_model_merge(args):
    return build_quant_merge(args)