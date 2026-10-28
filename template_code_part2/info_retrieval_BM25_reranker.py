from util import *

import math
from collections import Counter
from sentence_transformers import CrossEncoder


class InformationRetrieval():

    def __init__(self):

        self.index = None

        # Zero-shot reranker
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        )


    # -------------------------------------------------
    # BUILD BM25+ INDEX
    # -------------------------------------------------
    def buildIndex(self, docs, docIDs):

        total_docs = len(docIDs)

        doc_term_freq = {}
        doc_freq = {}
        doc_order = {}

        doc_lengths = {}
        total_doc_len = 0

        # store flattened document text
        raw_doc_texts = {}

        for position, docID in enumerate(docIDs):

            doc_order[docID] = position

            tokens = []

            for sentence in docs[position]:
                tokens.extend(sentence)

            raw_doc_texts[docID] = " ".join(tokens)

            term_counts = Counter(tokens)

            doc_term_freq[docID] = term_counts

            doc_len = sum(term_counts.values())
            doc_lengths[docID] = doc_len
            total_doc_len += doc_len

            for term in term_counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        avg_doc_len = total_doc_len / total_docs

        # BM25 IDF
        idf = {}

        for term, df in doc_freq.items():

            idf[term] = math.log(
                ((total_docs - df + 0.5) / (df + 0.5)) + 1.0
            )

        self.index = {
            "docIDs": list(docIDs),
            "doc_order": doc_order,
            "doc_term_freq": doc_term_freq,
            "idf": idf,
            "doc_lengths": doc_lengths,
            "avg_doc_len": avg_doc_len,
            "raw_doc_texts": raw_doc_texts
        }


    # -------------------------------------------------
    # BM25+ RETRIEVAL
    # -------------------------------------------------
    def bm25_retrieve(self, query_tokens, top_k=10):

        docIDs = self.index["docIDs"]
        doc_order = self.index["doc_order"]
        doc_term_freq = self.index["doc_term_freq"]
        idf = self.index["idf"]
        doc_lengths = self.index["doc_lengths"]
        avg_doc_len = self.index["avg_doc_len"]

        k1 = 1.5
        b = 0.75
        delta = 1.0

        query_tf = Counter(query_tokens)

        doc_scores = []

        for docID in docIDs:

            score = 0.0

            doc_tf = doc_term_freq[docID]
            doc_len = doc_lengths[docID]

            for term in query_tf:

                if term in doc_tf and term in idf:

                    tf = doc_tf[term]

                    numerator = tf * (k1 + 1)

                    denominator = tf + k1 * (
                        1 - b + b * (doc_len / avg_doc_len)
                    )

                    bm25_component = numerator / denominator

                    score += idf[term] * (
                        bm25_component + delta
                    )

            doc_scores.append(
                (score, doc_order[docID], docID)
            )

        doc_scores.sort(
            key=lambda item: (-item[0], item[1])
        )

        return [
            docID for _, _, docID in doc_scores[:top_k]
        ]


    # -------------------------------------------------
    # ZERO-SHOT RERANKING
    # -------------------------------------------------
    def rerank(self, query_tokens, candidate_docIDs):

        raw_doc_texts = self.index["raw_doc_texts"]

        query_text = " ".join(query_tokens)

        pairs = []

        for docID in candidate_docIDs:

            doc_text = raw_doc_texts[docID]

            pairs.append(
                (query_text, doc_text)
            )

        scores = self.reranker.predict(
            pairs,
            batch_size=32
        )

        reranked = []

        for score, docID in zip(scores, candidate_docIDs):
            reranked.append((score, docID))

        reranked.sort(reverse=True)

        return [docID for _, docID in reranked]


    # -------------------------------------------------
    # FINAL RANK PIPELINE
    # -------------------------------------------------
    def rank(self, queries):

        doc_IDs_ordered = []

        if self.index is None:
            return doc_IDs_ordered

        for query in queries:

            query_tokens = []

            for sentence in query:
                query_tokens.extend(sentence)

            # Stage 1: BM25+ retrieval
            candidate_docIDs = self.bm25_retrieve(
                query_tokens,
                top_k=10
            )

            # Stage 2: Neural reranking
            reranked_docIDs = self.rerank(
                query_tokens,
                candidate_docIDs
            )

            # append remaining docs
            remaining_docs = [
                docID
                for docID in self.index["docIDs"]
                if docID not in reranked_docIDs
            ]

            final_ranking = reranked_docIDs + remaining_docs

            doc_IDs_ordered.append(final_ranking)

        return doc_IDs_ordered
