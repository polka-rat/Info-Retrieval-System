import os
import time

import numpy as np
from gensim.models import KeyedVectors, Word2Vec


class InformationRetrieval:
    def __init__(
        self,
        vector_size=100,
        pretrained_path=r"C:\Users\Nagasai\OneDrive\Desktop\Acads\Sem 6\CS6370\Project\template_code_part2\GoogleNews-vectors-negative300.bin",
        embedding_path=None,
        use_pretrained=False,
        train_if_missing=True,
        temperature=8.0,
        window=5,
        min_count=1,
        workers=None,
    ):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.embedding_path = embedding_path or os.path.join(self.base_dir, "embeddings.kv")
        self.pretrained_path = pretrained_path
        self.use_pretrained = use_pretrained
        self.train_if_missing = train_if_missing
        self.vector_size = vector_size
        self.temperature = float(temperature)
        self.window = window
        self.min_count = min_count
        self.workers = workers if workers is not None else max(1, os.cpu_count() or 1)
        self.model = None
        self.embedding_source = None
        self.docIDs = []
        self.doc_order = {}
        self.doc_sentence_vectors = []
        self.doc_sentence_norms = []
    def _get_kv(self):
        if self.model is None:
            return None
        return self.model.wv if hasattr(self.model, "wv") else self.model
    def _embedding_source_for_path(self, path):
        if not path:
            return None
        abs_path = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if mtime is None:
            return f"file:{abs_path}"
        return f"file:{abs_path}:{mtime:.6f}"
    def _load_embeddings_from_path(self, path):
        if not path or not os.path.exists(path):
            return False

        print(f"[SentenceWiseW2V] Loading embeddings from: {path}")
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        if ext in {"bin", "vec", "txt", "gz"}:
            print(
                f"[SentenceWiseW2V] Detected raw word2vec format for {path}; "
                "using load_word2vec_format first"
            )
            binary = ext in {"bin", "gz"}
            try:
                self.model = KeyedVectors.load_word2vec_format(path, binary=binary)
            except Exception:
                print(
                    f"[SentenceWiseW2V] load_word2vec_format failed for {path}; "
                    "trying KeyedVectors.load"
                )
                try:
                    self.model = KeyedVectors.load(path)
                except Exception:
                    self.model = None
                    print(f"[SentenceWiseW2V] Failed to load embeddings from: {path}")
                    return False
        else:
            try:
                self.model = KeyedVectors.load(path)
            except Exception:
                try:
                    print(
                        f"[SentenceWiseW2V] KeyedVectors.load failed for {path}; "
                        "trying word2vec format"
                    )
                    binary = path.lower().endswith(".bin")
                    self.model = KeyedVectors.load_word2vec_format(path, binary=binary)
                except Exception:
                    self.model = None
                    print(f"[SentenceWiseW2V] Failed to load embeddings from: {path}")
                    return False

        kv = self._get_kv()
        if kv is not None and getattr(kv, "vector_size", None) is not None:
            self.vector_size = int(kv.vector_size)
        self.embedding_source = self._embedding_source_for_path(path)
        vocab_size = len(kv) if kv is not None else 0
        print(
            f"[SentenceWiseW2V] Loaded embeddings from: {path} | "
            f"vocab={vocab_size} dim={self.vector_size}"
        )
        return True

    def _train_embeddings(self, docs):
        sentences = []
        for doc in docs:
            for sentence in doc:
                if sentence:
                    sentences.append(sentence)

        if not sentences:
            raise ValueError("Cannot train Word2Vec embeddings on an empty corpus.")

        t0 = time.perf_counter()
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        self.model.wv.save(self.embedding_path)
        self.embedding_source = self._embedding_source_for_path(self.embedding_path)
        print(f"Word2Vec training time: {time.perf_counter() - t0:.4f}s")

    def _ensure_embeddings(self, docs):
        if self.model is not None:
            return
        candidate_paths = []
        if self.use_pretrained:
            if self.pretrained_path:
                candidate_paths.append(self.pretrained_path)
            candidate_paths.append(self.embedding_path)

        for path in candidate_paths:
            if path and self._load_embeddings_from_path(path):
                return

        if not self.train_if_missing:
            raise FileNotFoundError(
                "No pretrained Word2Vec model was found and train_if_missing=False."
            )
        print("[SentenceWiseW2V] No pretrained model found; training on the current corpus")
        self._train_embeddings(docs)

    def _word_vector(self, token):
        kv = self._get_kv()
        if kv is None:
            return None

        if token in kv:
            return kv[token]

        lowered = token.lower()
        if lowered in kv:
            return kv[lowered]
        return None



    def _sentence_vector(self, sentence):
        vectors = []
        for token in sentence:
            vec = self._word_vector(token)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    def _doc_sentence_matrix(self, doc):
        sentence_vectors = []
        sentence_norms = []

        for sentence in doc:
            vec = self._sentence_vector(sentence)
            sentence_vectors.append(vec)
            norm = float(np.linalg.norm(vec))
            sentence_norms.append(norm if norm > 0.0 else 1e-12)
        if not sentence_vectors:
            sentence_vectors = [np.zeros(self.vector_size, dtype=np.float32)]
            sentence_norms = [1e-12]
        return (
            np.asarray(sentence_vectors, dtype=np.float32),
            np.asarray(sentence_norms, dtype=np.float32),
        )

    def _query_sentence_vectors(self, query):
        vectors = []
        for sentence in query:
            vec = self._sentence_vector(sentence)
            if np.linalg.norm(vec) > 0.0:
                vectors.append(vec)
        return vectors

    def _cosine_scores(self, query_vector, doc_matrix, doc_norms):
        q_norm = float(np.linalg.norm(query_vector))
        if q_norm <= 0.0 or doc_matrix.size == 0:
            return np.zeros(0, dtype=np.float32)

        numerators = np.dot(doc_matrix, query_vector)
        denom = doc_norms * q_norm
        scores = np.zeros(len(numerators), dtype=np.float32)
        valid = denom > 0.0
        if np.any(valid):
            scores[valid] = numerators[valid] / denom[valid]
        return scores
    def _soft_pool(self, scores):
        if scores.size == 0:
            return 0.0

        scaled = scores * self.temperature
        max_scaled = float(np.max(scaled))
        mean_exp = float(np.mean(np.exp(scaled - max_scaled)))
        return (max_scaled + np.log(mean_exp)) / self.temperature

    def _score_document(self, query_sentence_vectors, doc_matrix, doc_norms):
        if not query_sentence_vectors:
            return 0.0
        per_sentence_scores = []
        for query_vector in query_sentence_vectors:
            sentence_scores = self._cosine_scores(query_vector, doc_matrix, doc_norms)
            if sentence_scores.size == 0:
                continue
            per_sentence_scores.append(self._soft_pool(sentence_scores))
        if not per_sentence_scores:
            return 0.0

        return float(np.mean(per_sentence_scores))




    def buildIndex(self, docs, docIDs):
        start = time.time()

        self.docIDs = list(docIDs)
        self.doc_order = {docID: position for position, docID in enumerate(self.docIDs)}
        self.doc_sentence_vectors = []
        self.doc_sentence_norms = []

        self._ensure_embeddings(docs)

        for doc in docs:
            doc_matrix, doc_norms = self._doc_sentence_matrix(doc)
            self.doc_sentence_vectors.append(doc_matrix)
            self.doc_sentence_norms.append(doc_norms)

        end = time.time()
        print(f"Sentence-wise index built in {end - start:.2f} seconds")

    def rank(self, queries):
        start = time.time()

        if not self.doc_sentence_vectors:
            return []

        if self.model is None:
            candidate_paths = []
            if self.use_pretrained:
                if self.pretrained_path:
                    candidate_paths.append(self.pretrained_path)
                candidate_paths.append(self.embedding_path)
            for path in candidate_paths:
                if path and self._load_embeddings_from_path(path):
                    break

        doc_IDs_ordered = []

        for query in queries:
            query_sentence_vectors = self._query_sentence_vectors(query)
            doc_scores = []

            for position, docID in enumerate(self.docIDs):
                score = self._score_document(
                    query_sentence_vectors,
                    self.doc_sentence_vectors[position],
                    self.doc_sentence_norms[position],
                )
                doc_scores.append((score, self.doc_order[docID], docID))

            doc_scores.sort(key=lambda item: (-item[0], item[1]))
            doc_IDs_ordered.append([docID for _, _, docID in doc_scores])

        end = time.time()
        print(f"Sentence-wise ranking completed for {len(queries)} queries in {end - start:.2f} seconds")
        return doc_IDs_ordered
