import math
import time
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


class InformationRetrieval:

    def __init__(
        self,
        mode="lsa",
        lsa_components=100,
        expansion_strength=0.6,
        top_k_neighbors=12,
        random_state=42,
        k=None,
        **_unused,
    ):
        if k is not None:
            lsa_components = k

        self.mode = "lsa"
        self.lsa_components = int(lsa_components)
        self.expansion_strength = float(expansion_strength)
        self.top_k_neighbors = int(top_k_neighbors)
        self.random_state = int(random_state)

        self.docIDs = []
        self.doc_order = {}
        self.index_to_word = []
        self.word_to_idx = {}
        self.vocab = self.word_to_idx
        self.idf = None
        self.doc_matrix = None
        self.doc_norms = None
        self.expansion_matrix = None
        self.index = None

    def _flatten_document(self, document):
        tokens = []
        for item in document:
            if isinstance(item, (list, tuple)):
                tokens.extend(item)
            elif item:
                tokens.append(item)
        return [token.lower() for token in tokens if token]

    def _build_tfidf_index(self, docs, docIDs):
        processed_docs = [self._flatten_document(doc) for doc in docs]
        doc_freq = Counter()
        for tokens in processed_docs:
            doc_freq.update(set(tokens))

        vocab_terms = list(doc_freq.keys())
        vocab_terms.sort(key=lambda term: (-doc_freq[term], term))

        self.index_to_word = vocab_terms
        self.word_to_idx = {word: idx for idx, word in enumerate(self.index_to_word)}
        self.vocab = self.word_to_idx

        total_docs = len(processed_docs)
        vocab_size = len(self.index_to_word)

        self.idf = np.zeros(vocab_size, dtype=float)
        for word, idx in self.word_to_idx.items():
            self.idf[idx] = math.log((total_docs + 1.0) / (doc_freq[word] + 1.0)) + 1.0

        rows = []
        cols = []
        data = []
        for doc_idx, tokens in enumerate(processed_docs):
            counts = Counter(token for token in tokens if token in self.word_to_idx)
            for term, freq in counts.items():
                term_idx = self.word_to_idx[term]
                weight = (1.0 + math.log(freq)) * self.idf[term_idx]
                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(weight)

        if total_docs == 0 or vocab_size == 0:
            self.doc_matrix = csr_matrix((total_docs, vocab_size), dtype=float)
            self.doc_norms = np.ones(total_docs, dtype=float)
            return

        doc_matrix = csr_matrix((data, (rows, cols)), shape=(total_docs, vocab_size), dtype=float)
        doc_norms = np.sqrt(doc_matrix.multiply(doc_matrix).sum(axis=1)).A1
        doc_norms[doc_norms == 0.0] = 1.0
        self.doc_matrix = diags(1.0 / doc_norms).dot(doc_matrix).tocsr()
        self.doc_norms = doc_norms

    def _latent_knn_matrix(self, embeddings):
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            return csr_matrix((0, 0), dtype=float)

        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms[:, np.newaxis]

        vocab_size = embeddings.shape[0]
        if vocab_size <= 1 or self.top_k_neighbors <= 0:
            return csr_matrix((vocab_size, vocab_size), dtype=float)

        n_neighbors = min(self.top_k_neighbors + 1, vocab_size)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        nn.fit(embeddings)
        distances, neighbors = nn.kneighbors(embeddings, return_distance=True)

        rows = []
        cols = []
        data = []
        for row_idx in range(vocab_size):
            candidate_scores = {}
            for dist, col_idx in zip(distances[row_idx], neighbors[row_idx]):
                col_idx = int(col_idx)
                if col_idx == row_idx:
                    continue
                score = 1.0 - float(dist)
                if score > 0.0 and score > candidate_scores.get(col_idx, 0.0):
                    candidate_scores[col_idx] = score

            if not candidate_scores:
                continue

            ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
            ranked = ranked[: self.top_k_neighbors]
            total = sum(score for _, score in ranked)
            if total <= 0.0:
                continue

            for col_idx, score in ranked:
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(score / total)

        return csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=float)

    def _build_lsa_matrix(self):
        vocab_size = len(self.index_to_word)
        if vocab_size == 0 or self.doc_matrix is None or self.doc_matrix.shape[0] == 0:
            return csr_matrix((vocab_size, vocab_size), dtype=float)

        max_rank = min(self.doc_matrix.shape[0], self.doc_matrix.shape[1])
        k = min(self.lsa_components, max_rank - 1) if max_rank > 1 else 0
        if k <= 0:
            return csr_matrix((vocab_size, vocab_size), dtype=float)

        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        svd.fit(self.doc_matrix)
        term_embeddings = svd.components_.T * svd.singular_values_
        return self._latent_knn_matrix(term_embeddings)

    def _query_raw_tfidf(self, query):
        q_vec = np.zeros(len(self.index_to_word), dtype=float)
        if not self.index_to_word:
            return q_vec

        tokens = self._flatten_document(query)
        counts = Counter(token for token in tokens if token in self.word_to_idx)
        for term, freq in counts.items():
            idx = self.word_to_idx[term]
            q_vec[idx] = (1.0 + math.log(freq)) * self.idf[idx]
        return q_vec

    def _expand_query_vector(self, raw_vector):
        if raw_vector.size == 0 or np.count_nonzero(raw_vector) == 0:
            return raw_vector

        if self.expansion_matrix is None or self.expansion_matrix.nnz == 0:
            expanded = raw_vector.copy()
        else:
            propagated = np.asarray(raw_vector @ self.expansion_matrix).ravel()
            expanded = raw_vector + (self.expansion_strength * propagated)

        norm = float(np.linalg.norm(expanded))
        if norm <= 0.0:
            return expanded
        return expanded / norm

    def buildIndex(self, docs, docIDs, *_, **__):
        start = time.time()

        self.docIDs = list(docIDs)
        self.doc_order = {doc_id: idx for idx, doc_id in enumerate(self.docIDs)}

        self._build_tfidf_index(docs, docIDs)
        self.expansion_matrix = self._build_lsa_matrix()

        self.index = {
            "mode": self.mode,
            "docIDs": self.docIDs,
            "doc_order": self.doc_order,
            "vocab": self.vocab,
            "idf": self.idf,
            "doc_matrix": self.doc_matrix,
            "expansion_matrix": self.expansion_matrix,
        }

        elapsed = time.time() - start
        print(f"[QE-LSA] Index built in {elapsed:.2f} seconds")

    def rank(self, queries, *_, **__):
        start = time.time()
        if self.doc_matrix is None:
            return []

        doc_IDs_ordered = []
        doc_order_indices = np.arange(len(self.docIDs))

        for query in queries:
            raw_query = self._query_raw_tfidf(query)
            expanded_query = self._expand_query_vector(raw_query)

            if expanded_query.size == 0 or np.count_nonzero(expanded_query) == 0:
                doc_IDs_ordered.append(list(self.docIDs))
                continue

            scores = np.asarray(self.doc_matrix.dot(expanded_query)).ravel()
            order = np.lexsort((doc_order_indices, -scores))
            doc_IDs_ordered.append([self.docIDs[idx] for idx in order])

        elapsed = time.time() - start
        print(f"[QE-LSA] Ranking completed for {len(queries)} queries in {elapsed:.2f} seconds")
        return doc_IDs_ordered
