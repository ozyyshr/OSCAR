#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# PYT=/home/ubuntu/anaconda3/envs/sharc_dpr/bin/python

for split in dev train
do
  python dense_retriever.py \
    --model_file=../sharc_data/DPR/dpr_biencoder.0.10945 \
    --ctx_file=../data/sharc_raw/id2snippet.json \
    --qa_file=../data/sharc_raw/json/sharc_${split}.json \
    --encoded_ctx_file=../sharc_data/DPR_quesce/_0.pkl \
    --out_file=../sharc_data/DPR_quesce/DPR_0_${split}.json \
    --n-docs=100
done