# Add your import statements here
import math
from collections import Counter

import numpy as np
from nltk.corpus import wordnet as wn


class InformationRetrieval():

	def __init__(self, retrieval_mode="tfidf"):
		self.index = None
		self.retrieval_mode = retrieval_mode

	def setRetrievalMode(self, retrieval_mode):
		"""
		Switch between supported retrieval strategies.

		Supported values:
		- "tfidf" (default)
		- "lsa_synset"
		"""
		self.retrieval_mode = retrieval_mode

	def flatten_document(self, document):
		tokens = []
		for sentence in document:
			tokens.extend(sentence)
		return tokens

	def concept_key_to_id(self, concept_key, concept_to_id, id_to_concept, allow_new):
		if concept_key not in concept_to_id:
			if not allow_new:
				return None
			concept_to_id[concept_key] = len(id_to_concept)
			id_to_concept.append(concept_key)
		return concept_to_id[concept_key]

	def token_to_concept_ids(self, token, concept_to_id, id_to_concept, allow_new=True):
		# Keep synsets as integer concept IDs internally instead of lemma/name strings.
		synsets = wn.synsets(token)
		if synsets:
			concept_keys = [synsets[0].name()]
		else:
			concept_keys = ["TOKEN::" + token]

		concept_ids = []
		for concept_key in concept_keys:
			concept_id = self.concept_key_to_id(
				concept_key,
				concept_to_id,
				id_to_concept,
				allow_new
			)
			if concept_id is not None:
				concept_ids.append(concept_id)
		return concept_ids

	def build_concept_counter(self, tokens, concept_to_id, id_to_concept, allow_new=True):
		concept_counts = Counter()
		for token in tokens:
			for concept_id in self.token_to_concept_ids(
				token,
				concept_to_id,
				id_to_concept,
				allow_new
			):
				concept_counts[concept_id] += 1
		return concept_counts

	def build_tfidf_index(self, docs, docIDs):
		total_docs = len(docIDs)
		doc_term_freq = {}
		doc_freq = {}
		doc_norms = {}
		doc_order = {}

		for position, docID in enumerate(docIDs):
			doc_order[docID] = position
			tokens = self.flatten_document(docs[position])
			term_counts = Counter(tokens)
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

		return {
			"mode": "tfidf",
			"docIDs": list(docIDs),
			"doc_order": doc_order,
			"doc_term_freq": doc_term_freq,
			"idf": idf,
			"doc_norms": doc_norms
		}

	def build_lsa_synset_index(self, docs, docIDs):
		total_docs = len(docIDs)
		doc_order = {}
		doc_concept_freq = {}
		doc_freq = Counter()
		concept_to_id = {}
		id_to_concept = []

		for position, docID in enumerate(docIDs):
			doc_order[docID] = position
			tokens = self.flatten_document(docs[position])
			concept_counts = self.build_concept_counter(
				tokens,
				concept_to_id,
				id_to_concept,
				allow_new=True
			)
			doc_concept_freq[docID] = concept_counts
			for concept_id in concept_counts:
				doc_freq[concept_id] += 1

		num_concepts = len(id_to_concept)
		concept_to_row = {concept_id: concept_id for concept_id in range(num_concepts)}

		idf = {}
		for concept_id, df in doc_freq.items():
			idf[concept_id] = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0

		matrix = np.zeros((num_concepts, total_docs), dtype=float)
		for col, docID in enumerate(docIDs):
			for concept_id, tf in doc_concept_freq[docID].items():
				row = concept_to_row[concept_id]
				matrix[row, col] = tf * idf[concept_id]

		if matrix.size == 0:
			return {
				"mode": "lsa_synset",
				"docIDs": list(docIDs),
				"doc_order": doc_order,
				"concept_to_id": concept_to_id,
				"id_to_concept": id_to_concept,
				"concept_to_row": concept_to_row,
				"idf": idf,
				"uk": None,
				"sk": None,
				"doc_vectors": np.zeros((total_docs, 0), dtype=float)
			}

		u, s, vt = np.linalg.svd(matrix, full_matrices=False)
		max_rank = min(matrix.shape[0], matrix.shape[1])
		k = min(100, max(1,max_rank-1))

		uk = u[:, :k]
		sk = s[:k]
		vtk = vt[:k, :]
		doc_vectors = (np.diag(sk).dot(vtk)).T

		return {
			"mode": "lsa_synset",
			"docIDs": list(docIDs),
			"doc_order": doc_order,
			"concept_to_id": concept_to_id,
			"id_to_concept": id_to_concept,
			"concept_to_row": concept_to_row,
			"idf": idf,
			"uk": uk,
			"sk": sk,
			"doc_vectors": doc_vectors
		}

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

		if self.retrieval_mode == "lsa_synset":
			self.index = self.build_lsa_synset_index(docs, docIDs)
		else:
			self.index = self.build_tfidf_index(docs, docIDs)

	def rank_tfidf(self, queries):
		doc_IDs_ordered = []

		docIDs = self.index["docIDs"]
		doc_order = self.index["doc_order"]
		doc_term_freq = self.index["doc_term_freq"]
		idf = self.index["idf"]
		doc_norms = self.index["doc_norms"]

		for query in queries:
			query_tokens = self.flatten_document(query)
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
				if query_norm > 0.0 and doc_norm > 0.0:
					score = dot_product / (query_norm * doc_norm)
				else:
					score = 0.0

				doc_scores.append((score, doc_order[docID], docID))

			doc_scores.sort(key=lambda item: (-item[0], item[1]))
			ranked_docIDs = [docID for _, _, docID in doc_scores]
			doc_IDs_ordered.append(ranked_docIDs)

		return doc_IDs_ordered

	def rank_lsa_synset(self, queries):
		doc_IDs_ordered = []

		docIDs = self.index["docIDs"]
		doc_order = self.index["doc_order"]
		concept_to_id = self.index["concept_to_id"]
		id_to_concept = self.index["id_to_concept"]
		concept_to_row = self.index["concept_to_row"]
		idf = self.index["idf"]
		uk = self.index["uk"]
		sk = self.index["sk"]
		doc_vectors = self.index["doc_vectors"]

		for query in queries:
			query_tokens = self.flatten_document(query)
			query_concepts = self.build_concept_counter(
				query_tokens,
				concept_to_id,
				id_to_concept,
				allow_new=False
			)
			query_vector = np.zeros(len(concept_to_row), dtype=float)

			for concept_id, tf in query_concepts.items():
				row = concept_to_row.get(concept_id)
				if row is not None:
					query_vector[row] = tf * idf[concept_id]

			if uk is None or sk is None or len(sk) == 0:
				doc_scores = [(0.0, doc_order[docID], docID) for docID in docIDs]
				doc_scores.sort(key=lambda item: (-item[0], item[1]))
				doc_IDs_ordered.append([docID for _, _, docID in doc_scores])
				continue

			safe_inverse = np.array([1.0 / value if value > 1e-12 else 0.0 for value in sk])
			query_latent = query_vector.dot(uk).dot(np.diag(safe_inverse))
			query_norm = np.linalg.norm(query_latent)

			doc_scores = []
			for idx, docID in enumerate(docIDs):
				doc_latent = doc_vectors[idx]
				doc_norm = np.linalg.norm(doc_latent)
				if query_norm > 0.0 and doc_norm > 0.0:
					score = float(np.dot(query_latent, doc_latent) / (query_norm * doc_norm))
				else:
					score = 0.0
				doc_scores.append((score, doc_order[docID], docID))

			doc_scores.sort(key=lambda item: (-item[0], item[1]))
			doc_IDs_ordered.append([docID for _, _, docID in doc_scores])

		return doc_IDs_ordered

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

		if self.index is None:
			return []

		if self.index["mode"] == "lsa_synset":
			return self.rank_lsa_synset(queries)
		return self.rank_tfidf(queries)
