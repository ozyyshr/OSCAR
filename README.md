# OSCAR

This repo contains codes for the following paper:

**Smoothing Dialogue States for Open Conversational Machine Reading.**

(Codes are being cleaned and updated in progress)

### 1. Retrieval
We use the linear combination of DPR and TF-IDF, the codes are based on [DrQA](https://github.com/facebookresearch/DrQA)
##### TF-IDF
The codes correspondingly are in `retriever_tfidf`. 
Firstly, we build the DB for documents:
```
python build_db.py
```
Then we compute the tf-idf scores based on the constructed DB:
```
python build_tfidf.py
```
Finally, we inference the candidates for each sample in ORShARC:
```
bash inference_tfidf.sh
```
##### DPR
