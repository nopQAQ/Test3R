#!/bin/bash

set -e

workdir='.'
model_name='DUSt3R_ViTLarge_BaseDecoder_512_dpt'
model_type='ours'
# model_type='dust3r'
output_dir="${workdir}/eval_results/mv_recon/${model_type}"
echo "$output_dir"

# Evaluation on 7Scenes
scenes=({0..17})
epoches=('1')
lrs=('0.00001')
accum_iters=( '2' )

for idx in "${scenes[@]}";do
    python -W ignore eval/mv_recon/launch.py  \
        --output_dir "${output_dir}" \
        --model_name "$model_name" \
        --model_type "$model_type" \
        --index "${idx}" \
        --epoches 1 \
        --lr 0.00001\
        --accum_iter 2 \
        --prompt 32     
done

# Evaluation on NRGBD
# scenes=({0..9})
# epoches=('1')
# lrs=('0.00001')
# accum_iters=( '2' )

# for idx in "${scenes[@]}";do
#     python -W ignore eval/mv_recon/launch.py  \
#         --output_dir "${output_dir}" \
#         --model_name "$model_name" \
#         --model_type "$model_type" \
#         --index "${idx}" \
#         --epoches 1 \
#         --lr 0.00008\
#         --accum_iter 2 \
#         --prompt 32     
# done