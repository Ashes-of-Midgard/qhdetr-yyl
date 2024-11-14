# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import torchvision.transforms as TcT


class CocoDetection(TvCocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    ):
        super(CocoDetection, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # scales = [704]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose([T.RandomResize([800], max_size=1333), normalize,])
        # return T.Compose([T.RandomResize([704], max_size=1333), normalize,])

    raise ValueError(f"unknown {image_set}")


###### PREDICTIONS MERGE MODIFIED ######
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    对经过归一化的张量进行反归一化操作。

    Args:
        tensor (torch.Tensor): 要反归一化的张量，形状为 (C, H, W)。
        mean (list): 归一化时使用的均值，默认为 [0.485, 0.456, 0.406]。
        std (list): 归一化时使用的标准差，默认为 [0.229, 0.224, 0.225]。

    Returns:
        torch.Tensor: 反归一化后的张量。
    """
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    tensor = tensor * std.view(-1, 1, 1)
    tensor = tensor + mean.view(-1, 1, 1)

    return tensor


def pil_transform_back(image_tensor):
    """
    将经过ToTensor和Normalize变换后的张量转换回PIL图像。

    Args:
        image_tensor (torch.Tensor): 经过变换的图像张量，形状为 (C, H, W)。

    Returns:
        PIL.Image.Image: 恢复后的PIL图像。
    """
    unnormalized_tensor = unnormalize(image_tensor)

    # 将张量的值范围从 [0, 1] 转换到 [0, 255] 并转换为无符号8位整数类型
    pil_image = TcT.ToPILImage()(unnormalized_tensor.clamp(0, 1).mul(255).byte())

    return pil_image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def draw_boxes_on_image(image, pred_bbox1=None, pred_bbox2=None, bbox1_c="b", bbox2_c="r", save_path=".draw_image.png"):
    """
    将给定的两个不同的边界框序列分别用不同颜色绘制到图像上，并保存到指定位置。

    Args:
        image (PIL.Image.Image): 要绘制边界框的PIL图像。
        pred_bbox1 (torch.Tensor): 第一个预测的边界框张量，形状为 [n_q, 4]，格式为 [x, y, w, h]，数值在 (0, 1) 表示图像尺寸的百分比。
        pred_bbox2 (torch.Tensor): 第二个预测的边界框张量，形状为 [n_q, 4]，格式为 [x, y, w, h]，数值在 (0, 1) 表示图像尺寸的百分比。
        save_path (str): 保存绘制好边界框图像的本地路径。

    Returns:
        None
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    width, height = image.size

    # 绘制第一个方框序列，用红色
    if pred_bbox1 is not None:
        for box in pred_bbox1:
            x, y, w, h = box.tolist()
            x_pixel = int(x * width)
            y_pixel = int(y * height)
            w_pixel = int(w * width)
            h_pixel = int(h * height)
            rect = patches.Rectangle((x_pixel, y_pixel), w_pixel, h_pixel, linewidth=1, edgecolor=bbox1_c, facecolor='none')
            ax.add_patch(rect)

    # 绘制第二个方框序列，用蓝色
    if pred_bbox2 is not None:
        for box in pred_bbox2:
            x, y, w, h = box.tolist()
            x_pixel = int(x * width)
            y_pixel = int(y * height)
            w_pixel = int(w * width)
            h_pixel = int(h * height)
            rect = patches.Rectangle((x_pixel, y_pixel), w_pixel, h_pixel, linewidth=1, edgecolor=bbox2_c, facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')

    # 保存图像到指定路径
    plt.savefig(save_path)
    plt.close()
###### END MODIFIED ######


def build(image_set, args, eval_in_training_set):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    if eval_in_training_set:
        image_set = "val"
        print("use validation dataset transforms")
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
    )
    return dataset
