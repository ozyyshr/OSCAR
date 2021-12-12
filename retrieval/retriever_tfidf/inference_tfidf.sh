#!/usr/bin/env bash

for split in train dev
do
  echo "=-=-=-=-=-=-=${split}-=-=-=-=-=-=-"
  python inference_tfidf.py \
    --qa_file=../data/sharc_raw/json/sharc_${split}.json \
    --db_path=../data/sharc_raw/id2snippet.json \
    --out_file=../sharc_data/tfidf/${split}.json \
    --tfidf_path=../data/sharc_raw/snippet-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
done