CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/qhdetr.sh \
    --coco_path ../datasets/coco2017 \
    --eval \
    --resume ./bestcheckpoint_0488.pth