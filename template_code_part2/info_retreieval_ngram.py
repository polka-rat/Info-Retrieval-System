from util import *
import math
from collections import Counter
import time

class InformationRetrieval():

	def __init__(self, n=4):
		self.index = None
		self.n = n   

	def _generate_ngrams(self, tokens):
		if self.n == 1:
			return tokens
		return [
			tuple(tokens[i:i+self.n]) 
			for i in range(len(tokens) - self.n + 1)
		]

	def buildIndex(self, docs, docIDs):
		start = time.time()
		index = None
		total_docs = len(docIDs)
		doc_term_freq = {}
		doc_freq = {}
		doc_norms = {}
		doc_order = {}
		for position, docID in enumerate(docIDs):
			doc_order[docID] = position
			tokens = []
			for sentence in docs[position]:
				tokens.extend(sentence)
			ngrams = self._generate_ngrams(tokens)
			term_counts = Counter(ngrams)
			doc_term_freq[docID] = term_counts
			for term in term_counts:
				doc_freq[term] = doc_freq.get(term, 0) + 1
		idf = {}
		for term, df in doc_freq.items():
			idf[term] = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0
		for docID in docIDs:
			norm_square = 0.0
			for term, tf in doc_term_freq[docID].items():
				weight = tf * idf[term]
				norm_square += weight * weight
			doc_norms[docID] = math.sqrt(norm_square)
		index = {
			"docIDs": list(docIDs),
			"doc_order": doc_order,
			"doc_term_freq": doc_term_freq,
			"idf": idf,
			"doc_norms": doc_norms
		}
		self.index = index
		end = time.time()
		print(f"Index built in {end - start:.2f} seconds")

	def rank(self, queries):
		start = time.time()
		doc_IDs_ordered = []
		if self.index is None:
			return doc_IDs_ordered
		docIDs = self.index["docIDs"]
		doc_order = self.index["doc_order"]
		doc_term_freq = self.index["doc_term_freq"]
		idf = self.index["idf"]
		doc_norms = self.index["doc_norms"]

		for query in queries:
			query_tokens = []
			for sentence in query:
				query_tokens.extend(sentence)

			query_ngrams = self._generate_ngrams(query_tokens)
			query_tf = Counter(query_ngrams)
			query_weights = {}
			query_norm_square = 0.0
			for term, tf in query_tf.items():
				if term in idf:
					weight = tf * idf[term]
					query_weights[term] = weight
					query_norm_square += weight * weight
			query_norm = math.sqrt(query_norm_square)
			doc_scores = []
			for docID in docIDs:
				dot_product = 0.0
				doc_tf = doc_term_freq[docID]
				for term, query_weight in query_weights.items():
					if term in doc_tf:
						doc_weight = doc_tf[term] * idf[term]
						dot_product += query_weight * doc_weight
				doc_norm = doc_norms[docID]
				if query_norm > 0.0 and doc_norm > 0.0:
					score = dot_product / (query_norm * doc_norm)
				else:
					score = 0.0
				doc_scores.append((score, doc_order[docID], docID))
			doc_scores.sort(key=lambda item: (-item[0], item[1]))
			ranked_docIDs = [docID for _, _, docID in doc_scores]
			doc_IDs_ordered.append(ranked_docIDs)
		end = time.time()
		print(f"Ranking completed for {len(queries)} queries in {end - start:.2f} seconds")
		return doc_IDs_ordered