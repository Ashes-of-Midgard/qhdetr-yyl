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
