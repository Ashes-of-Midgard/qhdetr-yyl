#!/usr/bin/env bash

set -x

EXP_DIR=exps/12eps/300-1500
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --epochs 12 \
    --lr_drop 11 \
    --weight_decay 0.0 \
    --num_queries_one2one 15 \
    --num_queries_one2many 75 \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --quant \
    --predictions_merge \
    --lowest_number_predictions_one2one 3 \
    --lowest_number_predictions_one2many 13 \
    ${PY_ARGS}
