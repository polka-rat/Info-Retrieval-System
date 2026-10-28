import numpy as np
from collections import Counter
import math
import lightgbm as lgb


class LTRRanker:

    def __init__(self):
        self.model = None
        self.idf = None
        self.doc_term_freq = None
        self.doc_lengths = None
        self.avg_doc_len = None
        self.docIDs = None
        self.total_docs = 0

    # -------------------------
    # BUILD STATS
    # -------------------------
    def build_stats(self, docs, docIDs):

        self.docIDs = docIDs
        self.total_docs = len(docIDs)

        doc_term_freq = {}
        doc_freq = {}
        doc_lengths = {}

        total_len = 0

        for i, docID in enumerate(docIDs):
            tokens = []
            for sent in docs[i]:
                tokens.extend(sent)

            tf = Counter(tokens)
            doc_term_freq[docID] = tf

            length = sum(tf.values())
            doc_lengths[docID] = length
            total_len += length

            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self.avg_doc_len = total_len / self.total_docs

        # IDF (BM25 style)
        idf = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

        self.idf = idf
        self.doc_term_freq = doc_term_freq
        self.doc_lengths = doc_lengths

    # -------------------------
    # BM25 (CORE FEATURE)
    # -------------------------
    def bm25(self, query_tokens, docID, k1=1.5, b=0.75):

        score = 0
        doc_tf = self.doc_term_freq[docID]
        doc_len = self.doc_lengths[docID]

        for term in query_tokens:
            if term in doc_tf and term in self.idf:
                tf = doc_tf[term]
                idf = self.idf[term]

                denom = tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                score += idf * (tf * (k1 + 1)) / denom

        return score

    # -------------------------
    # LEXICAL FEATURES (IMPORTANT 5)
    # -------------------------
    def extract_features(self, query_tokens, docID):

        doc_tf = self.doc_term_freq[docID]
        query_tf = Counter(query_tokens)

        # 1. BM25 (main ranking signal)
        bm25_score = self.bm25(query_tokens, docID)

        # 2. Query term coverage (recall proxy)
        matched_terms = sum(1 for t in query_tf if t in doc_tf)
        coverage = matched_terms / (len(query_tf) + 1e-6)

        # 3. Term frequency match strength
        tf_sum = sum(doc_tf.get(t, 0) for t in query_tf)

        # 4. Document length penalty (normalized)
        doc_len = self.doc_lengths[docID]
        norm_len = doc_len / self.avg_doc_len

        # 5. Query frequency in document (importance signal)
        q_freq = sum(doc_tf.get(t, 0) for t in query_tf)

        return [
            bm25_score,
            coverage,
            tf_sum,
            norm_len,
            q_freq
        ]

    # -------------------------
    # DATASET PREPARATION (FIXED RELEVANCE)
    # -------------------------
    def prepare_dataset(self, queries, query_ids, qrels):

        X, y, group = [], [], []

        rel_dict = {}
        for item in qrels:
            qid = int(item["query_num"])
            docid = int(item["id"])

            rel = int(item["position"])
            rel = min(rel, 2)

            if qid not in rel_dict:
                rel_dict[qid] = {}
            rel_dict[qid][docid] = rel

        for i, query in enumerate(queries):

            qid = query_ids[i]

            tokens = []
            for sent in query:
                tokens.extend(sent)

            count = 0

            for docID in self.docIDs:

                features = self.extract_features(tokens, docID)
                relevance = rel_dict.get(qid, {}).get(docID, 0)

                X.append(features)
                y.append(relevance)
                count += 1

            group.append(count)

        return np.array(X), np.array(y), group

    # -------------------------
    # TRAIN
    # -------------------------
    def train(self, X, y, group):

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31
        )

        self.model.fit(X, y, group=group)

    # -------------------------
    # RANK
    # -------------------------
    def rank(self, queries):

        results = []

        for query in queries:

            tokens = []
            for sent in query:
                tokens.extend(sent)

            scores = []

            for docID in self.docIDs:
                feat = self.extract_features(tokens, docID)
                score = self.model.predict([feat])[0]
                scores.append((score, docID))

            scores.sort(reverse=True)
            results.append([d for _, d in scores])

        return results