import math
from collections import Counter

import numpy as np


class InformationRetrieval():
    def __init__(self, lsa_weight=0.3, tfidf_weight=0.7, k=100):
        self.lsa_weight = lsa_weight
        self.tfidf_weight = tfidf_weight
        self.k = k

        self.docIDs = []
        self.doc_order = {}

        self.vocab = {}
        self.idf = {}
        self.doc_term_freq = {}
        self.doc_norms = {}

        self.Uk = None
        self.Sk = None
        self.doc_vectors_lsa = None

    def flatten_document(self, document):
        tokens = []
        for sentence in document:
            tokens.extend(sentence)
        return tokens

    def buildIndex(self, docs, docIDs):
        self.docIDs = list(docIDs)
        self.doc_order = {doc_id: idx for idx, doc_id in enumerate(self.docIDs)}
        total_docs = len(self.docIDs)

        processed_docs = []
        vocab_set = set()
        self.doc_term_freq = {}

        for i, doc_id in enumerate(self.docIDs):
            tokens = self.flatten_document(docs[i])
            processed_docs.append(tokens)
            vocab_set.update(tokens)
            self.doc_term_freq[doc_id] = Counter(tokens)

        self.vocab = {term: i for i, term in enumerate(vocab_set)}

        doc_freq = Counter()
        for doc_id in self.docIDs:
            for term in self.doc_term_freq[doc_id]:
                doc_freq[term] += 1

        self.idf = {}
        for term, df in doc_freq.items():
            self.idf[term] = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0

        self.doc_norms = {}
        for doc_id in self.docIDs:
            norm_sq = 0.0
            for term, tf in self.doc_term_freq[doc_id].items():
                w = tf * self.idf[term]
                norm_sq += w * w
            self.doc_norms[doc_id] = math.sqrt(norm_sq)

        v_size = len(self.vocab)
        tfidf_matrix = np.zeros((v_size, total_docs), dtype=float)

        for col, doc_id in enumerate(self.docIDs):
            for term, tf in self.doc_term_freq[doc_id].items():
                row = self.vocab[term]
                tfidf_matrix[row, col] = tf * self.idf[term]

        if tfidf_matrix.size == 0:
            self.Uk = None
            self.Sk = None
            self.doc_vectors_lsa = np.zeros((total_docs, 0), dtype=float)
            return

        U, S, Vt = np.linalg.svd(tfidf_matrix, full_matrices=False)
        max_rank = min(tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        rank_k = min(self.k, max_rank) if max_rank > 0 else 0

        if rank_k == 0:
            self.Uk = None
            self.Sk = None
            self.doc_vectors_lsa = np.zeros((total_docs, 0), dtype=float)
            return

        self.Uk = U[:, :rank_k]
        self.Sk = S[:rank_k]
        self.doc_vectors_lsa = Vt[:rank_k, :].T

    def _query_tfidf_weights(self, query_tokens):
        q_tf = Counter(query_tokens)
        q_weights = {}
        q_norm_sq = 0.0

        for term, tf in q_tf.items():
            if term in self.idf:
                w = tf * self.idf[term]
                q_weights[term] = w
                q_norm_sq += w * w

        return q_weights, math.sqrt(q_norm_sq)

    def _tfidf_similarity(self, q_weights, q_norm, doc_id):
        if q_norm == 0.0 or self.doc_norms.get(doc_id, 0.0) == 0.0:
            return 0.0

        dot = 0.0
        doc_tf = self.doc_term_freq[doc_id]
        for term, q_w in q_weights.items():
            if term in doc_tf:
                dot += q_w * (doc_tf[term] * self.idf[term])

        return dot / (q_norm * self.doc_norms[doc_id])

    def _lsa_query_vector(self, query_tokens):
        if self.Uk is None or self.Sk is None or len(self.Sk) == 0:
            return None

        q_vec = np.zeros(len(self.vocab), dtype=float)
        q_tf = Counter(query_tokens)
        for term, tf in q_tf.items():
            if term in self.vocab and term in self.idf:
                q_vec[self.vocab[term]] = tf * self.idf[term]

        safe_inverse = np.array([1.0 / s if s > 1e-12 else 0.0 for s in self.Sk])
        q_latent = q_vec.dot(self.Uk).dot(np.diag(safe_inverse))
        return q_latent

    def _lsa_similarity(self, q_latent, doc_idx):
        if q_latent is None:
            return 0.0
        d_latent = self.doc_vectors_lsa[doc_idx]
        q_norm = np.linalg.norm(q_latent)
        d_norm = np.linalg.norm(d_latent)
        if q_norm == 0.0 or d_norm == 0.0:
            return 0.0
        return float(np.dot(q_latent, d_latent) / (q_norm * d_norm))

    def rank(self, queries):
        if not self.docIDs:
            return []

        doc_IDs_ordered = []
        for query in queries:
            query_tokens = self.flatten_document(query)

            q_weights, q_norm = self._query_tfidf_weights(query_tokens)
            q_latent = self._lsa_query_vector(query_tokens)

            scored_docs = []
            for doc_idx, doc_id in enumerate(self.docIDs):
                tfidf_sim = self._tfidf_similarity(q_weights, q_norm, doc_id)
                lsa_sim = self._lsa_similarity(q_latent, doc_idx)
                hybrid_score = (self.lsa_weight * lsa_sim) + (self.tfidf_weight * tfidf_sim)
                scored_docs.append((hybrid_score, self.doc_order[doc_id], doc_id))

            scored_docs.sort(key=lambda item: (-item[0], item[1]))
            doc_IDs_ordered.append([doc_id for _, _, doc_id in scored_docs])

        return doc_IDs_ordered
