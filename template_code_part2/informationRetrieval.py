from util import *

# Add your import statements here
import math
from collections import Counter




class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		total_docs = len(docIDs)
		doc_term_freq = {}
		doc_freq = {}
		doc_norms = {}
		doc_order = {}

		for position, docID in enumerate(docIDs):
			# Preserve input order so ties are resolved deterministically.
			doc_order[docID] = position
			tokens = []
			# Collapse sentence-level tokens into a single document token stream.
			for sentence in docs[position]:
				tokens.extend(sentence)

			term_counts = Counter(tokens)
			doc_term_freq[docID] = term_counts

			for term in term_counts:
				doc_freq[term] = doc_freq.get(term, 0) + 1

		idf = {}
		# Compute global IDF for all terms seen in the corpus.
		for term, df in doc_freq.items():
			# Smoothed IDF keeps values finite and non-zero for seen terms.
			idf[term] = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0

		# Precompute document vector norms for cosine normalization at query time.
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


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

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
			# Use the same flattening strategy for queries and documents.
			for sentence in query:
				query_tokens.extend(sentence)

			query_tf = Counter(query_tokens)
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
				# Cosine similarity between query and document TF-IDF vectors.
				if query_norm > 0.0 and doc_norm > 0.0:
					score = dot_product / (query_norm * doc_norm)
				else:
					score = 0.0

				doc_scores.append((score, doc_order[docID], docID))

			# Rank by score descending, then by original document order.
			doc_scores.sort(key=lambda item: (-item[0], item[1]))
			ranked_docIDs = [docID for _, _, docID in doc_scores]
			doc_IDs_ordered.append(ranked_docIDs)
	
		return doc_IDs_ordered




