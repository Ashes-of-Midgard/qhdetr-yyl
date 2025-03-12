# QHDETR

## install

```shell
# CUDA 11.8
conda create -n qhdetr python=3.8 -y
conda activate qhdetr
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv<2.2.0,>=2.0.0rc4"
mim install mmdet
git clone https://github.com/Ashes-of-Midgard/qhdetr-yyl.git
cd qhdetr-yyl/models/ops
python setup.py build install
```

## prepare dataset&pretrained weights

### 数据集
#### COCO2017
在qhdetr-yyl的上一级目录创建目录```datasets```
从官方网站下载COCO2017数据集，目录结构为：
```
- datasets
    - coco2017
        - train2017
            - xxx.jpg
        - val2017
            - xxx.jpg
        - annotations
            - instances_train2017.json
            - instances_val2017.json
```

### 模型权重
在qhdetr-yyl的上一级目录创建目录```models```
下载模型权重：
resnet50-4bit: 通过网盘分享的文件：rest50-4bit-7346.pth
链接: https://pan.baidu.com/s/1J7LIKKlNuE9_v0Vedyrsvg?pwd=r32d 提取码: r32d 
--来自百度网盘超级会员v6的分享
deformable_detr：通过网盘分享的文件：r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth
链接: https://pan.baidu.com/s/1hI394aO3RLa_-vixfnLvRQ?pwd=6cdm 提取码: 6cdm 
--来自百度网盘超级会员v6的分享
存储于```models```目录下

## script

训练
```shell
bash train.sh
```
根据卡数调整学习率，默认配置是8卡，学习率lr=2e-4, lr_backbone=2e-5，根据卡数线性调整。通过在```train.sh```里传入python命令行参数调整

参考（调整为4卡）：
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/qhdetr.sh \
    --coco_path ../datasets/coco2017 \
    --resume ../models/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
    --load_q_RN50 \
    --batch_size 1 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
```
