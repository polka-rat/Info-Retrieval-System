from util import *

# Add your import statements here
import math
from collections import Counter


class InformationRetrieval():

	def __init__(self):
		self.index = None


	def buildIndex(self, docs, docIDs):
		"""
		Builds BM25+ index statistics
		"""

		total_docs = len(docIDs)

		doc_term_freq = {}
		doc_freq = {}
		doc_order = {}

		doc_lengths = {}
		total_doc_len = 0

		# -------------------------
		# BUILD TERM FREQUENCIES
		# -------------------------
		for position, docID in enumerate(docIDs):

			doc_order[docID] = position

			tokens = []

			for sentence in docs[position]:
				tokens.extend(sentence)

			term_counts = Counter(tokens)

			doc_term_freq[docID] = term_counts

			# document length
			doc_len = sum(term_counts.values())
			doc_lengths[docID] = doc_len
			total_doc_len += doc_len

			# document frequency
			for term in term_counts:
				doc_freq[term] = doc_freq.get(term, 0) + 1

		# -------------------------
		# AVERAGE DOCUMENT LENGTH
		# -------------------------
		avg_doc_len = total_doc_len / total_docs

		# -------------------------
		# BM25 IDF
		# -------------------------
		idf = {}

		for term, df in doc_freq.items():

			idf[term] = math.log(
				((total_docs - df + 0.5) / (df + 0.5)) + 1.0
			)

		# -------------------------
		# STORE INDEX
		# -------------------------
		self.index = {

			"docIDs": list(docIDs),

			"doc_order": doc_order,

			"doc_term_freq": doc_term_freq,

			"idf": idf,

			"doc_lengths": doc_lengths,

			"avg_doc_len": avg_doc_len
		}


	def rank(self, queries):
		"""
		Rank documents using BM25+
		"""

		doc_IDs_ordered = []

		if self.index is None:
			return doc_IDs_ordered

		# -------------------------
		# LOAD INDEX
		# -------------------------
		docIDs = self.index["docIDs"]

		doc_order = self.index["doc_order"]

		doc_term_freq = self.index["doc_term_freq"]

		idf = self.index["idf"]

		doc_lengths = self.index["doc_lengths"]

		avg_doc_len = self.index["avg_doc_len"]

		# -------------------------
		# BM25+ PARAMETERS
		# -------------------------
		k1 = 1.5
		b = 0.75
		delta = 1.0

		# -------------------------
		# PROCESS EACH QUERY
		# -------------------------
		for query in queries:

			query_tokens = []

			for sentence in query:
				query_tokens.extend(sentence)

			query_tf = Counter(query_tokens)

			doc_scores = []

			# -------------------------
			# SCORE EACH DOCUMENT
			# -------------------------
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

			# -------------------------
			# SORT DOCUMENTS
			# -------------------------
			doc_scores.sort(
				key=lambda item: (-item[0], item[1])
			)

			ranked_docIDs = [
				docID for _, _, docID in doc_scores
			]

			doc_IDs_ordered.append(ranked_docIDs)

		return doc_IDs_ordered