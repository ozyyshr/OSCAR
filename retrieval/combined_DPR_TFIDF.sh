#!/usr/bin/env bash

for split in dev train
do
  echo "=-=-=-=-=-=-=${split}-=-=-=-=-=-=-"
  python combined_DPR_TFIDF.py \
    --qa_file=./data/sharc_raw/json/sharc_${split}.json \
    --db_path=./data/sharc_raw/id2snippet.json \
    --tfidf_path=./sharc_data/tfidf/${split}.json \
    --dpr_path=./sharc_data/DPR_quesce/DPR_0_${split}.json \
    --out_file=./sharc_data/combined/${split}.json \
    --combined_weight=2
done