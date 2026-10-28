import numpy as np
from collections import Counter
import math
import lightgbm as lgb


class LTRRanker:

    def __init__(self):

        self.model = None

        # TF-IDF statistics
        self.idf = None
        self.doc_term_freq = None
        self.doc_norms = None

        # Metadata
        self.docIDs = None
        self.total_docs = 0

    # ---------------------------------------------------
    # BUILD TF-IDF STATISTICS
    # ---------------------------------------------------
    def build_stats(self, docs, docIDs):

        self.docIDs = docIDs

        self.total_docs = len(docIDs)

        doc_term_freq = {}

        doc_freq = {}

        # -----------------------------------------------
        # Build TF and DF
        # -----------------------------------------------
        for i, docID in enumerate(docIDs):

            tokens = []

            for sentence in docs[i]:
                tokens.extend(sentence)

            tf = Counter(tokens)

            doc_term_freq[docID] = tf

            for term in tf:

                doc_freq[term] = (
                    doc_freq.get(term, 0) + 1
                )

        # -----------------------------------------------
        # Compute IDF
        # -----------------------------------------------
        idf = {}

        for term, df in doc_freq.items():

            idf[term] = math.log(
                (self.total_docs + 1.0) /
                (df + 1.0)
            ) + 1.0

        # -----------------------------------------------
        # Compute document norms
        # -----------------------------------------------
        doc_norms = {}

        for docID in docIDs:

            norm_sq = 0.0

            for term, tf in doc_term_freq[docID].items():

                weight = tf * idf[term]

                norm_sq += weight * weight

            doc_norms[docID] = math.sqrt(norm_sq)

        self.idf = idf

        self.doc_term_freq = doc_term_freq

        self.doc_norms = doc_norms

    # ---------------------------------------------------
    # COSINE SIMILARITY
    # ---------------------------------------------------
    def cosine_similarity(self, query_tokens, docID):

        doc_tf = self.doc_term_freq[docID]

        query_tf = Counter(query_tokens)

        dot_product = 0.0

        query_norm_sq = 0.0

        for term, q_tf in query_tf.items():

            if term not in self.idf:
                continue

            q_weight = q_tf * self.idf[term]

            query_norm_sq += q_weight * q_weight

            if term in doc_tf:

                d_weight = (
                    doc_tf[term] *
                    self.idf[term]
                )

                dot_product += (
                    q_weight * d_weight
                )

        query_norm = math.sqrt(query_norm_sq)

        doc_norm = self.doc_norms[docID]

        if query_norm > 0 and doc_norm > 0:

            cosine = (
                dot_product /
                (query_norm * doc_norm)
            )

        else:
            cosine = 0.0

        return cosine

    # ---------------------------------------------------
    # FEATURE EXTRACTION
    # ONLY 3 FEATURES
    # ---------------------------------------------------
    def extract_features(self, query_tokens, docID):

        doc_tf = self.doc_term_freq[docID]

        query_tf = Counter(query_tokens)

        # -----------------------------------------------
        # 1. TF-IDF score
        # -----------------------------------------------
        tfidf_score = 0.0

        for term, q_tf in query_tf.items():

            if (
                term in self.idf and
                term in doc_tf
            ):

                q_weight = (
                    q_tf * self.idf[term]
                )

                d_weight = (
                    doc_tf[term] *
                    self.idf[term]
                )

                tfidf_score += (
                    q_weight * d_weight
                )

        # -----------------------------------------------
        # 2. Cosine similarity
        # -----------------------------------------------
        cosine = self.cosine_similarity(
            query_tokens,
            docID
        )

        # -----------------------------------------------
        # 3. Query term overlap
        # -----------------------------------------------
        overlap = len(
            set(query_tf.keys()) &
            set(doc_tf.keys())
        )

        return [
            tfidf_score,
            cosine,
            overlap
        ]

    # ---------------------------------------------------
    # PREPARE TRAINING DATASET
    # ---------------------------------------------------
    def prepare_dataset(
        self,
        queries,
        query_ids,
        qrels
    ):

        X = []

        y = []

        group = []

        # -----------------------------------------------
        # Relevance dictionary
        # -----------------------------------------------
        rel_dict = {}

        for item in qrels:

            qid = int(item["query_num"])

            docid = int(item["id"])

            pos = int(item["position"])

            # Correct Cranfield relevance mapping
            if pos == 1:
                rel = 3

            elif pos <= 3:
                rel = 2

            else:
                rel = 1

            if qid not in rel_dict:
                rel_dict[qid] = {}

            rel_dict[qid][docid] = rel

        # -----------------------------------------------
        # Build training examples
        # -----------------------------------------------
        for i, query in enumerate(queries):

            qid = query_ids[i]

            query_tokens = []

            for sentence in query:
                query_tokens.extend(sentence)

            count = 0

            for docID in self.docIDs:

                features = self.extract_features(
                    query_tokens,
                    docID
                )

                relevance = rel_dict.get(
                    qid,
                    {}
                ).get(docID, 0)

                X.append(features)

                y.append(relevance)

                count += 1

            group.append(count)

        return (
            np.array(X),
            np.array(y),
            group
        )

    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------
    def train(self, X, y, group):

        self.model = lgb.LGBMRanker(

            objective="lambdarank",

            metric="ndcg",

            n_estimators=150,

            learning_rate=0.05,

            num_leaves=31,

            random_state=42
        )

        self.model.fit(
            X,
            y,
            group=group
        )

    # ---------------------------------------------------
    # RANK DOCUMENTS
    # ---------------------------------------------------
    def rank(self, queries):

        results = []

        for query in queries:

            query_tokens = []

            for sentence in query:
                query_tokens.extend(sentence)

            scores = []

            for docID in self.docIDs:

                features = self.extract_features(
                    query_tokens,
                    docID
                )

                score = self.model.predict(
                    [features]
                )[0]

                scores.append(
                    (score, docID)
                )

            # Sort descending by score
            scores.sort(
                key=lambda x: x[0],
                reverse=True
            )

            ranked_docs = [
                docID
                for score, docID in scores
            ]

            results.append(ranked_docs)

        return results