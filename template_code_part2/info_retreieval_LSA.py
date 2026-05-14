import math
from collections import Counter, defaultdict
import numpy as np
import time


class InformationRetrieval():
    def __init__(self, k=100):
        self.vocab = {}                
        self.idf = None
        self.U = None
        self.S = None
        self.Vt = None
        self.doc_vectors = None
        self.docIDs = []
        self.k = k

    def buildIndex(self, docs, docIDs, context_length=1):
        """
        Build LSA model from documents
        """
        start = time.time()
        self.docIDs = list(docIDs)

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
                TDM[i, j] = 1 + np.log(freq)
        df = np.count_nonzero(TDM > 0, axis=1)
        self.idf = np.log((D + 1) / (df + 1)) + 1
        TFIDF = TDM * self.idf[:, np.newaxis]
        TFIDF /= (np.linalg.norm(TFIDF, axis=0, keepdims=True) + 1e-8)
        U, S, Vt = np.linalg.svd(TFIDF, full_matrices=False)
        actual_k = min(self.k, len(S))
        
        
        print(f"Original dimensionality: {len(S)}, Reduced dimensionality: {actual_k}")
        
        
        
        self.U = U[:, :actual_k]
        self.S = S[:actual_k]
        self.Vt = Vt[:actual_k, :]
        self.doc_vectors = (np.diag(self.S) @ self.Vt).T
        end = time.time()
        print(f"Index built in {end - start:.2f} seconds")

    def rank(self, queries):

        start = time.time()
        doc_IDs_ordered = []
        doc_norms = np.linalg.norm(self.doc_vectors, axis=1) + 1e-8
        S_inv = np.diag(1 / (self.S + 1e-8))
        for query in queries:
            tokens = []
            for sent in query:
                tokens.extend(sent)
            q_vec = np.zeros(len(self.vocab))
            counts = Counter(tokens)
            for word, freq in counts.items():
                if word in self.vocab:
                    q_vec[self.vocab[word]] = 1 + np.log(freq)

            q_vec = q_vec * self.idf
            q_vec /= (np.linalg.norm(q_vec) + 1e-8)

            q_latent = q_vec @ self.U @ S_inv

            q_norm = np.linalg.norm(q_latent) + 1e-8

            scores = self.doc_vectors @ q_latent
            scores /= (doc_norms * q_norm)

            ranked = np.argsort(scores)[::-1]

            doc_IDs_ordered.append([self.docIDs[idx] for idx in ranked])

        end = time.time()

        print(f"Ranking completed for {len(queries)} queries in {end - start:.2f} seconds")

        return doc_IDs_ordered
