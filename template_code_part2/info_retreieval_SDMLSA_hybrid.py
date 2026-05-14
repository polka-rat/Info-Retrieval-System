import math
import time
from collections import Counter
from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn


class InformationRetrieval():
    def __init__(self, k=100):
        self.vocab = {}
        self.docIDs = []
        self.doc_order = {}
        self.idf = None
        self.U = None
        self.S = None
        self.Vt = None
        self.doc_vectors = None
        self.k = k
        self.context_length = 1

    @lru_cache(maxsize=None)
    def get_synsets(self, word):
        return wn.synsets(word)

    def get_best_synset(self, word, context):
        synsets = self.get_synsets(word)
        if not synsets:
            # Keep OOV tokens as their own pseudo-synset so they are not dropped.
            return word.lower()

        context_synsets_list = [self.get_synsets(w) for w in context]

        best_syn = synsets[0]
        best_score = -1

        for syn in synsets:
            score = 0
            for ctx_synsets in context_synsets_list:
                if not ctx_synsets:
                    continue

                max_sim = 0
                for ctx_syn in ctx_synsets:
                    sim = syn.wup_similarity(ctx_syn)
                    if sim and sim > max_sim:
                        max_sim = sim

                score += max_sim

            if score > best_score:
                best_score = score
                best_syn = syn

        return best_syn.name()

    def buildIndex(self, docs, docIDs, context_length=1):
        start = time.time()
        self.context_length = context_length
        self.docIDs = list(docIDs)
        self.doc_order = {docID: position for position, docID in enumerate(self.docIDs)}

        processed_docs = []
        vocab_set = set()
        for doc in docs:
            syn_tokens = []
            for sent in doc:
                for i in range(len(sent)):
                    token = sent[i]
                    context = sent[max(0, i - context_length):i] + sent[i + 1:i + 1 + context_length]
                    syn = self.get_best_synset(token, context)
                    if syn:
                        syn_tokens.append(syn)
                        vocab_set.add(syn)
            processed_docs.append(syn_tokens)

        self.vocab = {syn: i for i, syn in enumerate(vocab_set)}
        V = len(self.vocab)
        D = len(processed_docs)

        SDM = np.zeros((V, D))
        for j, syn_tokens in enumerate(processed_docs):
            counts = Counter(syn_tokens)
            for syn, freq in counts.items():
                i = self.vocab[syn]
                SDM[i, j] = freq

        df = np.count_nonzero(SDM > 0, axis=1)
        self.idf = np.log((D + 1) / (df + 1))
        TFIDF = SDM * self.idf[:, np.newaxis]
        U, S, Vt = np.linalg.svd(TFIDF, full_matrices=False)

        self.U = U[:, :self.k]
        self.S = S[:self.k]
        self.Vt = Vt[:self.k, :]
        self.doc_vectors = self.Vt.T

        end = time.time()
        print(f"Index built in {end - start:.2f} seconds")

    def rank(self, queries):
        start = time.time()
        doc_IDs_ordered = []

        for query in queries:
            syn_tokens = []
            for sent in query:
                for i in range(len(sent)):
                    token = sent[i]
                    context = sent[max(0, i - self.context_length):i] + sent[i + 1:i + 1 + self.context_length]
                    syn = self.get_best_synset(token, context)
                    if syn:
                        syn_tokens.append(syn)

            q_vec = np.zeros(len(self.vocab))
            counts = Counter(syn_tokens)

            for syn, freq in counts.items():
                if syn in self.vocab:
                    q_vec[self.vocab[syn]] = freq

            q_vec = q_vec * self.idf
            q_latent = np.dot(q_vec, self.U / self.S)

            scores = []
            for doc_vec in self.doc_vectors:
                num = np.dot(q_latent, doc_vec)
                denom = (np.linalg.norm(q_latent) * np.linalg.norm(doc_vec) + 1e-8)
                scores.append(num / denom)

            ranked = np.argsort(scores)[::-1]
            doc_IDs_ordered.append([self.docIDs[idx] for idx in ranked])

        end = time.time()
        print(f"Ranking completed for {len(queries)} queries in {end - start:.2f} seconds")
        return doc_IDs_ordered
