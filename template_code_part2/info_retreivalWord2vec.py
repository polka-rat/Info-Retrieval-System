import os
import math
import time
import numpy as np
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors


class InformationRetrieval:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.model = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.embedding_path = os.path.join(self.base_dir, "embeddings.kv")
        self.centroid_path = os.path.join(self.base_dir, "centroid.npy")
        self.tfidf_centroid_path = os.path.join(self.base_dir, "tfidf_centroid.npy")
        self.doc_ids_path = os.path.join(self.base_dir, "docIDs.npy")

        self.docIDs = []
        self.id_to_pos = {}

        self.idf = {}
        self.doc_freq = Counter()
        self.N = 0

        self.doc_vectors = {
            "centroid": None,
            "tfidf_centroid": None
        }

    def _flatten_doc(self, doc):
        tokens = []
        for sent in doc:
            tokens.extend(sent)
        return tokens

    def _get_kv(self):
        if self.model is None:
            return None
        return self.model.wv if hasattr(self.model, "wv") else self.model

    def train_embeddings(self, docs, save_path=None, min_count=2, window=5, workers=4):
        t0 = time.perf_counter()
        sentences = [sent for doc in docs for sent in doc]

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )

        if save_path:
            self.model.wv.save(save_path)
        print(f"Embedding model build/save time: {time.perf_counter() - t0:.4f}s")

    def load_embeddings(self, path):
        t0 = time.perf_counter()
        self.model = KeyedVectors.load(path)
        print(f"Embedding model load time: {time.perf_counter() - t0:.4f}s")

    def _ensure_embeddings(self, docs):
        if self.model is not None:
            return
        if os.path.exists(self.embedding_path):
            self.load_embeddings(self.embedding_path)
        else:
            self.train_embeddings(docs, save_path=self.embedding_path)

    def _can_load_saved_doc_vectors(self, docIDs):
        if not (
            os.path.exists(self.centroid_path)
            and os.path.exists(self.tfidf_centroid_path)
            and os.path.exists(self.doc_ids_path)
        ):
            return False
        try:
            saved_doc_ids = list(np.load(self.doc_ids_path, allow_pickle=True))
        except Exception:
            return False
        return saved_doc_ids == list(docIDs)

    def _build_idf(self, docs):
        """
        Build IDF over raw tokens.
        """
        self.doc_freq = Counter()
        self.N = len(docs)

        for doc in docs:
            tokens = set(self._flatten_doc(doc))
            for tok in tokens:
                self.doc_freq[tok] += 1

        self.idf = {}
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.N + 1.0) / (df + 1.0)) + 1.0

    def _word_vector(self, word):
        kv = self._get_kv()
        if kv is None:
            return None

        if word in kv:
            return kv[word]
        if word.lower() in kv:
            return kv[word.lower()]
        return None

    def _doc_centroid(self, tokens):
        """
        Simple average of word vectors.
        """
        vecs = []
        for word in tokens:
            v = self._word_vector(word)
            if v is not None:
                vecs.append(v)

        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)

        return np.mean(np.asarray(vecs, dtype=np.float32), axis=0)

    def _doc_tfidf_centroid(self, tokens):
        """
        TF-IDF weighted centroid of word vectors.
        """
        tf = Counter(tokens)
        weighted_vecs = []
        total_weight = 0.0

        for word, freq in tf.items():
            v = self._word_vector(word)
            if v is None:
                continue

            idf = self.idf.get(word, self.idf.get(word.lower(), 0.0))
            weight = float(freq) * float(idf)

            if weight > 0:
                weighted_vecs.append(weight * v)
                total_weight += weight

        if not weighted_vecs or total_weight == 0.0:
            return np.zeros(self.vector_size, dtype=np.float32)

        return np.sum(np.asarray(weighted_vecs, dtype=np.float32), axis=0) / total_weight

    def buildIndex(self, docs, docIDs):
        """
        Build both document representations:
        - centroid
        - tfidf_centroid
        """
        self.docIDs = list(docIDs)
        self.id_to_pos = {doc_id: i for i, doc_id in enumerate(self.docIDs)}

        self._ensure_embeddings(docs)
        self._build_idf(docs)

        if self._can_load_saved_doc_vectors(docIDs):
            t0 = time.perf_counter()
            self.doc_vectors["centroid"] = np.load(self.centroid_path, allow_pickle=True)
            print(f"Centroid load time: {time.perf_counter() - t0:.4f}s")
            t1 = time.perf_counter()
            self.doc_vectors["tfidf_centroid"] = np.load(self.tfidf_centroid_path, allow_pickle=True)
            print(f"TF-IDF centroid load time: {time.perf_counter() - t1:.4f}s")
            return

        centroid_vectors = []
        t0 = time.perf_counter()
        for doc in docs:
            tokens = self._flatten_doc(doc)
            centroid_vectors.append(self._doc_centroid(tokens))
        self.doc_vectors["centroid"] = np.asarray(centroid_vectors, dtype=np.float32)
        print(f"Centroid build time: {time.perf_counter() - t0:.4f}s")

        tfidf_centroid_vectors = []
        t1 = time.perf_counter()
        for doc in docs:
            tokens = self._flatten_doc(doc)
            tfidf_centroid_vectors.append(self._doc_tfidf_centroid(tokens))
        self.doc_vectors["tfidf_centroid"] = np.asarray(tfidf_centroid_vectors, dtype=np.float32)
        print(f"TF-IDF centroid build time: {time.perf_counter() - t1:.4f}s")

        np.save(self.centroid_path, self.doc_vectors["centroid"])
        np.save(self.tfidf_centroid_path, self.doc_vectors["tfidf_centroid"])
        np.save(self.doc_ids_path, np.array(self.docIDs, dtype=object))

    def save_centroids(self, directory):
        """
        Save both document-vector matrices as two different files:
        - centroid.npy
        - tfidf_centroid.npy
        """
        os.makedirs(directory, exist_ok=True)

        if self.doc_vectors["centroid"] is None:
            raise ValueError("Centroid vectors are not built yet.")
        if self.doc_vectors["tfidf_centroid"] is None:
            raise ValueError("TF-IDF centroid vectors are not built yet.")

        np.save(os.path.join(directory, "centroid.npy"), self.doc_vectors["centroid"])
        np.save(os.path.join(directory, "tfidf_centroid.npy"), self.doc_vectors["tfidf_centroid"])

        np.save(os.path.join(directory, "docIDs.npy"), np.array(self.docIDs, dtype=object))

    def load_centroids(self, directory, mode="centroid"):
        """
        Load one representation or both.

        mode:
            - 'centroid'
            - 'tfidf_centroid'
            - 'both'
        """
        if mode == "centroid":
            self.doc_vectors["centroid"] = np.load(os.path.join(directory, "centroid.npy"), allow_pickle=True)
            doc_ids_path = os.path.join(directory, "docIDs.npy")
            if os.path.exists(doc_ids_path):
                self.docIDs = list(np.load(doc_ids_path, allow_pickle=True))

        elif mode == "tfidf_centroid":
            self.doc_vectors["tfidf_centroid"] = np.load(os.path.join(directory, "tfidf_centroid.npy"), allow_pickle=True)
            doc_ids_path = os.path.join(directory, "docIDs.npy")
            if os.path.exists(doc_ids_path):
                self.docIDs = list(np.load(doc_ids_path, allow_pickle=True))

        elif mode == "both":
            self.doc_vectors["centroid"] = np.load(os.path.join(directory, "centroid.npy"), allow_pickle=True)
            self.doc_vectors["tfidf_centroid"] = np.load(os.path.join(directory, "tfidf_centroid.npy"), allow_pickle=True)
            doc_ids_path = os.path.join(directory, "docIDs.npy")
            if os.path.exists(doc_ids_path):
                self.docIDs = list(np.load(doc_ids_path, allow_pickle=True))
        else:
            raise ValueError("mode must be 'centroid', 'tfidf_centroid', or 'both'.")

        self.id_to_pos = {doc_id: i for i, doc_id in enumerate(self.docIDs)}

    def _vectorize_query(self, query, mode="centroid"):
        tokens = self._flatten_doc(query)

        if mode == "centroid":
            return self._doc_centroid(tokens)
        elif mode == "tfidf_centroid":
            return self._doc_tfidf_centroid(tokens)
        else:
            raise ValueError("mode must be either 'centroid' or 'tfidf_centroid'.")

    def rank(self, queries, mode="tfidf_centroid"):
        """
        Rank documents using cosine similarity.

        mode:
            - 'centroid'
            - 'tfidf_centroid'
        """
        if self.model is None and os.path.exists(self.embedding_path):
            self.load_embeddings(self.embedding_path)

        if mode not in self.doc_vectors or self.doc_vectors[mode] is None:
            raise ValueError(f"Document vectors not available for mode='{mode}'.")

        t0 = time.perf_counter()
        doc_matrix = self.doc_vectors[mode]
        doc_IDs_ordered = []

        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        doc_norms[doc_norms == 0.0] = 1e-12

        for query in queries:
            q_vec = self._vectorize_query(query, mode=mode)
            q_norm = np.linalg.norm(q_vec)

            if q_norm == 0.0:
                doc_IDs_ordered.append([])
                continue

            scores = np.dot(doc_matrix, q_vec) / (doc_norms * q_norm)
            ranked_idx = np.argsort(scores)[::-1]
            ranked_docIDs = [self.docIDs[i] for i in ranked_idx]
            doc_IDs_ordered.append(ranked_docIDs)

        print(f"Ranking time ({mode}): {time.perf_counter() - t0:.4f}s")
        return doc_IDs_ordered
