import math
from collections import Counter, defaultdict
import numpy as np
import time


class InformationRetrieval():
    def __init__(self, retrieval_mode="lsa", k=100):
        self.vocab = {}                # word → index
        self.idf = None
        self.U = None
        self.S = None
        self.Vt = None
        self.doc_vectors = None
        self.k = k                     

    def buildIndex(self, docs, docIDs, context_length=1):
        """
        Build LSA model from documents
        """
        start = time.time()
        processed_docs = []
        vocab_set = set()
        for doc in docs:
            tokens = []
            for sent in doc:
                tokens.extend(sent)
            processed_docs.append(tokens)
            vocab_set.update(tokens)

        self.vocab = {word: i for i, word in enumerate(vocab_set)}
        V = len(self.vocab)
        D = len(processed_docs)
        TDM = np.zeros((V, D))

        for j, tokens in enumerate(processed_docs):
            counts = Counter(tokens)
            for word, freq in counts.items():
                i = self.vocab[word]
                TDM[i, j] = freq

        df = np.count_nonzero(TDM > 0, axis=1)
        self.idf = np.log((D + 1) / (df + 1))

        TFIDF = TDM * self.idf[:, np.newaxis]

        U, S, Vt = np.linalg.svd(TFIDF, full_matrices=False)

        self.U = U[:, :self.k]
        self.S = S[:self.k]
        self.Vt = Vt[:self.k, :]

        self.doc_vectors = self.Vt.T
        end = time.time()
        print(f"Index built in {end - start:.2f} seconds")

    def rank(self, queries):
        """
        Rank documents using cosine similarity in latent space
        """
        start = time.time()

        doc_IDs_ordered = []
        for query in queries:
            tokens = []
            for sent in query:
                tokens.extend(sent)
            q_vec = np.zeros(len(self.vocab))
            counts = Counter(tokens)

            for word, freq in counts.items():
                if word in self.vocab:
                    q_vec[self.vocab[word]] = freq
            q_vec = q_vec * self.idf
            q_latent = np.dot(q_vec, self.U @ np.diag(1 / self.S)) 
            scores = []
            for i, doc_vec in enumerate(self.doc_vectors):
                num = np.dot(q_latent, doc_vec)
                denom = (np.linalg.norm(q_latent) * np.linalg.norm(doc_vec) + 1e-8)
                score = num / denom
                scores.append(score)

            # --- Step 6: Rank ---
            ranked = np.argsort(scores)[::-1]
            doc_IDs_ordered.append(ranked.tolist())
        end = time.time()
        print(f"Ranking completed for {len(queries)} queries in {end - start:.2f} seconds")
        return doc_IDs_ordered