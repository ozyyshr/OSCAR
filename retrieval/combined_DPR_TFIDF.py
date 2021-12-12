import sys
import json
import argparse
#sys.path.append('/private/home/sewonmin/EfficientQA-baselines/DPR')
#from dense_retriever import validate, save_results
# import drqa_retriever as retriever


def _loads_json(loadpath):
        with open(loadpath, 'r', encoding='utf-8') as fh:
            dataset = []
            for line in fh:
                example = json.loads(line)
                dataset.append(example)
        return dataset

def main(args):
    # questions = []
    # question_answers = []

    with open(args.qa_file) as f:
        qa_file = json.load(f)
    # qa_file = _loads_json(args.qa_file)

    # all_passages = load_passages(args.db_path)
    with open(args.db_path) as f:
        id2snippet = json.load(f)
    
    with open(args.tfidf_path) as f:
        tfidf_doc_score = json.load(f)
    
    with open(args.dpr_path, "r", encoding="utf-8") as f:
        dpr_doc_score = json.load(f)
    
    assert len(tfidf_doc_score) == len(dpr_doc_score)

    combined_doc_score = []
    for idx, ex in enumerate(qa_file):
        combined = {}
        tfidf = tfidf_doc_score[idx]
        assert tfidf[0] != []
        dpr = dpr_doc_score[idx]
        # if len(dpr[1]) == 1 and dpr[1][0]== 100:
        #     combined_doc_score.append(dpr)
        # else:
        for i, item in enumerate(tfidf[0]):
            combined[item] = tfidf[1][i]
        for j, item_ in enumerate(dpr[0]):
            if item_ in combined.keys():
                combined[item_] = dpr[1][j] * args.combined_weight + combined[item_]
            else:
                combined[item_] = dpr[1][j]
        sort = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        docs = []
        scores = []
        for pair in sort:
            docs.append(pair[0])
            scores.append(pair[1])
        combined_doc_score.append((docs, scores))

    top_ids_and_scores = combined_doc_score

    # validate
    matches = []
    for ex, top_psg_ids in zip(qa_file, top_ids_and_scores):
        # temp_matches = []
        # for curr_id in top_psg_ids[0]:
        #     if len(top_psg_ids[1]) == 1 and top_psg_ids[1][0] == 100:
        #         temp_matches.append(1)
        #     else:
        #         temp_matches.append(id2snippet[curr_id] == ex['snippet'])
        # matches.append(temp_matches)
        matches.append([id2snippet[curr_id] == ex['snippet'] for curr_id in top_psg_ids[0]])

    for top_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 100]:
        count = sum([any(curr_match[:top_n]) for curr_match in matches])
        print("Top {}: {:.1f}".format(top_n, count / len(matches) * 100))

    with open(args.out_file, 'w') as f:
        json.dump(top_ids_and_scores, f)

    # if len(all_passages) == 0:
    #     raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    # questions_doc_hits = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
    #                               args.match)
    #
    # if args.out_file:
    #     save_results(all_passages, questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', required=True, type=str, default=None)
    parser.add_argument('--dpr_path', type=str, default="../DPR")
    parser.add_argument('--db_path', type=str, default="/checkpoint/sewonmin/dpr/data/wikipedia_split/psgs_w100_seen_only.tsv")
    parser.add_argument('--tfidf_path', type=str, default="/checkpoint/sewonmin/dpr/drqa_retrieval_seen_only/db-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'])
    parser.add_argument('--n-docs', type=int, default=100)
    parser.add_argument('--validation_workers', type=int, default=16)
    parser.add_argument('--combined_weight', type=float, default=1)
    args = parser.parse_args()

    sys.path.append(args.dpr_path)
    # from dense_retriever import parse_qa_csv_file, load_passages, validate, save_results

    # ranker = retriever.get_class('tfidf')(tfidf_path=args.tfidf_path)

    main(args)


