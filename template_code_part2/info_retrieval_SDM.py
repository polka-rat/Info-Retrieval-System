import math
from collections import Counter
import numpy as np
from nltk.corpus import wordnet as wn
from functools import lru_cache
import time 


class InformationRetrieval():
    def __init__(self, retrieval_mode="tfidf"):
        self.index = {}
        self.df = Counter()              # document frequency
        self.doc_lengths = {}            # doc length normalization
        self.N = 0                       # total docs
        self.context_length = 1

    @lru_cache(maxsize=None)
    def get_synsets(self, word):
        return wn.synsets(word)
    
    def get_best_synset(self, word, context):
        synsets = self.get_synsets(word)
        if not synsets:
            return None

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
        self.context_length = context_length   # (5)
        self.N = len(docIDs)

        for doc, docID in zip(docs, docIDs):
            doc_len = 0

            for sent in doc:
                for i in range(len(sent)):
                    token = sent[i]
                    context = sent[max(0, i - context_length):i] + sent[i + 1:i + 1 + context_length]
                    best_syn = self.get_best_synset(token, context)
                    if best_syn:
                        doc_len += 1
                        if best_syn not in self.index:
                            self.index[best_syn] = {}
                        self.index[best_syn][docID] = self.index[best_syn].get(docID, 0) + 1
            self.doc_lengths[docID] = doc_len
        for syn in self.index:
            self.df[syn] = len(self.index[syn])
        end = time.time()
        print(f"Index built in {end - start:.2f} seconds")


    def rank(self, queries):
        start = time.time()
        doc_IDs_ordered = []
        for query in queries:
            doc_scores = {}
            for sent in query:
                for i in range(len(sent)):
                    token = sent[i]
                    context = sent[max(0, i - self.context_length):i] + sent[i + 1:i + 1 + self.context_length]
                    best_syn = self.get_best_synset(token, context)
                    if best_syn and best_syn in self.index:
                        idf = math.log((self.N + 1) / (self.df[best_syn] + 1))
                        for docID, freq in self.index[best_syn].items():
                            norm_tf = freq / self.doc_lengths[docID]
                            doc_scores[docID] = doc_scores.get(docID, 0) + norm_tf * idf
            ranked_docs = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
            doc_IDs_ordered.append(ranked_docs)
        end = time.time()
        print(f"Ranking completed for {len(queries)} queries in {end - start:.2f} seconds")
        return doc_IDs_ordered