# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

from .quant_backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
)
from .quant_dtransformer import build_deforamble_transformer
from .lsq_plus import *
from ._quan_base_plus import *
import copy
import pdb

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        num_queries_one2one=300,
        num_queries_one2many=0,
        mixed_selection=False,
        # ====== YYL MODIFIED - PREDICTIONS MERGE ======
        predictions_merge=False,
        lowest_number_predictions = 50,
        kl_div_threshold = 3e-7,
        iou_threshold = 0.8
        # ====== END MODIFIED - PREDICTIONS MERGE ======
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            num_queries_one2one: number of object queries for one-to-one matching part
            num_queries_one2many: number of object queries for one-to-many matching part
            mixed_selection: a trick for Deformable DETR two stage

        """
        super().__init__()
        # pdb.set_trace()
        num_queries = num_queries_one2one + num_queries_one2many
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes) # full precision
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # full precision
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1)
            if two_stage
            else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection
        # ====== YYL MODIFIED - PREDICTIONS MERGE ======
        self.predictions_merge = predictions_merge
        self.lowest_number_predictions = lowest_number_predictions
        self.kl_div_threshold = kl_div_threshold
        self.iou_threshold = iou_threshold
        # ====== END MODIFIED - PREDICTIONS MERGE ======

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # samples: NestedTensor, samples.tensors.shape=[batch_size, 3, H, W], samples.mask.shape=[batch_size, H, W]
        features, pos = self.backbone(samples)
        # features: List[NestedTensor]
        #   - features[i].tensor: shape=[batch_size, d_i, h_i, w_i]
        #   - features[i].mask: shape=[batch_size, h_i, w_i]
        # pos: List[Tensor]
        #   - pos[i]: shape=[batch_size, d_model, h_i, w_i]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            # self.input_proj: ModuleList
            #   - self.input_proj[l]: Sequence, d_l -> d_model
            # self.input_proj[l](src): Tensor, shape=[batch_size, d_model, h_l, w_l]
            srcs.append(self.input_proj[l](src))
            # mask: Tensor, shape=[batch_size, h_l, w_l]
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            # If the expected feature levels' number is larger than backbone's
            # output levels' number. The feature of the last level will be put
            # into a projection sequence, of which the output of each projection
            # will be used to fill the srcs list until expected length.
            #
            # features[-1].tensors -> input_proj[l]
            #                               |
            #                               V
            #                             src[l]
            #                               |
            #                               V
            #                         input_proj[l+1]
            #                               |
            #                               V
            #                             src[l+1]
            #                               |
            #                               V 
            #                             ......
            #                               |
            #                               V 
            #                             src[num_feature_levels - 1]
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                # masks are simply resized to the same size as features
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                # Use the backbone's position encoder to generate new pos embeddings for extra levels of feature
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]
            # query_embeds: Tensor, shape=[num_queries, d_model] or [num_queries, d_model*2]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros([self.num_queries, self.num_queries,]).bool().to(src.device)
        )
        self_attn_mask[self.num_queries_one2one :, 0 : self.num_queries_one2one,] = True
        self_attn_mask[0 : self.num_queries_one2one, self.num_queries_one2one :,] = True
        # self_attn_mask: Tensor, shape=[num_queries, num_queries]
        # note: num_queries==num_queries_one2one + num_queries_one2many
        
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        # hs: shape=[n_levels, batch_size, n_queries, d_model]
        # n_queries == n_queries_one2one + n_queries_one2many
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # reference: shape=[batch_size, n_queries, 4]
            # self.class_embed: ModuleList[Linear], containing: Linear(d_model, n_classes)
            outputs_class = self.class_embed[lvl](hs[lvl])
            # self.bbox_embed: ModuleList[MLP], containing: MLP(d_model, d_model, 4, 3)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            # outputs_coord: shape=[batch_size, n_queries, 4]
            # outputs_class: shape=[batch_size, n_queries, n_classes]

            outputs_classes_one2one.append(
                outputs_class[:, 0 : self.num_queries_one2one]
            )
            outputs_classes_one2many.append(
                outputs_class[:, self.num_queries_one2one :]
            )
            outputs_coords_one2one.append(
                outputs_coord[:, 0 : self.num_queries_one2one]
            )
            outputs_coords_one2many.append(
                outputs_coord[:, self.num_queries_one2one :]
            )
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        # outputs_classes_one2one: shape=[n_levels, batch_size, n_queries_one2one, n_classes]
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        # outputs_coords_one2one: shape=[n_levels, batch_size, n_queries_one2one, 4]
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        # outputs_classes_one2many: shape=[n_levels, batch_size, n_queries_one2many, n_classes]
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        # outputs_coords_one2many: shape=[n_levels, batch_size, n_queries_one2many, 4]

        # ====== YYL MODIFIED - PREDICTIONS MERGE ======
        if self.predictions_merge and self.training:
            outputs_classes = torch.concat([outputs_classes_one2one, outputs_classes_one2many], dim=2)
            outputs_coords = torch.concat([outputs_coords_one2one, outputs_coords_one2many], dim=2)
            del outputs_classes_one2one
            del outputs_classes_one2many
            del outputs_coords_one2one
            del outputs_coords_one2many
            torch.cuda.empty_cache()

            outputs_device = outputs_classes.device

            n_lvls, bs, n_q, n_cls = outputs_classes.shape
            with torch.no_grad():
                classes_index = outputs_classes.argmax(dim=3)
                classes_same_matrix = (classes_index.unsqueeze(2) == classes_index.unsqueeze(3))
                del classes_index
                torch.cuda.empty_cache()

                box_area = outputs_coords[:, :, :, 2] * outputs_coords[:, :, :, 3]

                X_min = (outputs_coords[:, :, :, 0] - 0.5*outputs_coords[:, :, :, 2]).detach()
                Y_min = (outputs_coords[:, :, :, 1] - 0.5*outputs_coords[:, :, :, 3]).detach()
                X_max = (outputs_coords[:, :, :, 0] + 0.5*outputs_coords[:, :, :, 2]).detach()
                Y_max = (outputs_coords[:, :, :, 1] + 0.5*outputs_coords[:, :, :, 3]).detach()
                
                X_intersection_min = torch.max(X_min.unsqueeze(2), X_min.unsqueeze(3))
                Y_intersection_min = torch.max(Y_min.unsqueeze(2), Y_min.unsqueeze(3))
                X_intersection_max = torch.min(X_max.unsqueeze(2), X_max.unsqueeze(3))
                Y_intersection_max = torch.min(Y_max.unsqueeze(2), Y_max.unsqueeze(3))
                del X_min
                del Y_min
                del X_max
                del Y_max
                torch.cuda.empty_cache()

                intersection_width = torch.max(torch.zeros([n_lvls, bs, n_q, n_q]).to(outputs_device), X_intersection_max - X_intersection_min)
                intersection_height = torch.max(torch.zeros([n_lvls, bs, n_q, n_q]).to(outputs_device), Y_intersection_max - Y_intersection_min)
                del X_intersection_min
                del Y_intersection_min
                del X_intersection_max
                del Y_intersection_max
                torch.cuda.empty_cache()

                intersection_area = intersection_width * intersection_height
                del intersection_width
                del intersection_height
                torch.cuda.empty_cache()

                union_area = (box_area.unsqueeze(2) + box_area.unsqueeze(3) - intersection_area)
                iou_matrix = intersection_area / (union_area + 1e-6)
                del box_area
                del union_area
                del intersection_area
                torch.cuda.empty_cache()

                merge_mask = classes_same_matrix & (iou_matrix > torch.tensor(self.iou_threshold))
                triangular_matrix = (torch.arange(n_q).unsqueeze(0) >= torch.arange(n_q).unsqueeze(1))[None, None]
                merge_mask = merge_mask & triangular_matrix.to(outputs_device)
                del classes_same_matrix
                del iou_matrix
                del triangular_matrix
                torch.cuda.empty_cache()
                
                # merge_mask: shape=[n_levels, batch_size, n_queries, n_queries]
                eye = torch.eye(n_q, dtype=torch.bool).to(outputs_device)[None, None]
                for i in range(n_q):
                    merge_mask = (~(merge_mask ^ eye)[:, :, i, :].unsqueeze(3)) & merge_mask
                outputs_valid_mask = merge_mask.any(dim=3)
                merge_mask = merge_mask.to(torch.float)
            
            # max_num_valid = torch.max(outputs_valid_mask.sum(dim=2).flatten()).item()
            # print(max_num_valid)
            outputs_classes_merged = torch.matmul(merge_mask, outputs_classes) / (merge_mask.sum(dim=3, keepdim=True)+1e-6)
            outputs_coords_merged = torch.matmul(merge_mask, outputs_coords) / (merge_mask.sum(dim=3, keepdim=True)+1e-6)

            del merge_mask
            del outputs_classes
            del outputs_coords
            torch.cuda.empty_cache()
            
            out = {
                "valid_mask": outputs_valid_mask[-1],
                "pred_logits": outputs_classes_merged[-1],
                "pred_boxes": outputs_coords_merged[-1],
            }
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_classes_merged, outputs_coords_merged, outputs_valid_mask
                )
        else:
            out = {
                "pred_logits": outputs_classes_one2one[-1],
                "pred_boxes": outputs_coords_one2one[-1],
                "pred_logits_one2many": outputs_classes_one2many[-1],
                "pred_boxes_one2many": outputs_coords_one2many[-1],
            }
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_classes_one2one, outputs_coords_one2one
                )
                out["aux_outputs_one2many"] = self._set_aux_loss(
                    outputs_classes_one2many, outputs_coords_one2many
                )
        # ====== END MODIFIED - PREDICTIONS MERGE ======

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused

    # ====== YYL MODIFIED - PREDICTIONS MERGE ======
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_valid_mask=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_valid_mask is None:
            return [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_boxes": b, "valid_mask": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_valid_mask[:-1])
            ]
    # ====== END MODIFIED - PREDICTIONS MERGE ======

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        # ====== YYL MODIFIED - PREDICTIONS MERGE ======
        if "valid_mask" in outputs.keys():
            valid_mask = outputs["valid_mask"]
        else:
            valid_mask = None
        # ====== YYL MODIFIED - PREDICTIONS MERGE ======

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
                # ====== YYL MODIFIED - PREDICTIONS MERGE ======
                valid_mask=valid_mask
                # ====== END MODIFIED - PREDICTIONS MERGE ======
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        # pdb.set_trace()
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, topk=100):
        super().__init__()
        self.topk = topk
        print("topk for eval:", self.topk)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.topk, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(
        #     nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        # )
        self.layers = nn.ModuleList(LinearLSQ(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        # pdb.set_trace()
        self.layers[-1] = nn.Linear(256, 4, bias=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_quant(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
        # ====== YYL MODIFIED - PREDICTIONS MERGE ======
        predictions_merge=args.predictions_merge,
        lowest_number_predictions=args.lowest_number_predictions
        # ====== END MODIFIED - PREDICTIONS MERGE ======
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_one2many"] = value
    weight_dict = new_dict

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha
    )
    criterion.to(device)
    postprocessors = {"bbox": PostProcess(topk=args.topk)}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85
            )

    return model, criterion, postprocessors
