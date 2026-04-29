import math
import re
from collections import Counter

import numpy as np
from nltk.corpus import wordnet as wn


class InformationRetrieval():

	def __init__(self, retrieval_mode="wsd_tfidf"):
		self.index = None
		self.retrieval_mode = retrieval_mode
		self.synset_signature_cache = {}
		self.synset_lookup_cache = {}
		self.context_window_size = 4

	def setRetrievalMode(self, retrieval_mode):
		"""
		Switch between supported retrieval strategies.

		Supported values:
		- "wsd_tfidf" (default)
		- "tfidf"
		- "lsa_synset"
		- "bm25"
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

	def normalize_text(self, text):
		return re.findall(r"[a-z0-9]+", text.lower())

	def normalize_token(self, token):
		parts = self.normalize_text(token)
		if not parts:
			return ""
		return " ".join(parts)

	def get_candidate_synsets(self, token):
		normalized_token = self.normalize_token(token)
		if not normalized_token:
			return []

		if normalized_token in self.synset_lookup_cache:
			return self.synset_lookup_cache[normalized_token]

		candidate_forms = [normalized_token]
		morph_form = wn.morphy(normalized_token)
		if morph_form and morph_form not in candidate_forms:
			candidate_forms.append(morph_form)

		synsets = []
		for form in candidate_forms:
			noun_synsets = wn.synsets(form, pos=wn.NOUN)
			if noun_synsets:
				synsets = noun_synsets
				break
			general_synsets = wn.synsets(form)
			if general_synsets:
				synsets = general_synsets
				break

		self.synset_lookup_cache[normalized_token] = synsets
		return synsets

	def synset_signature(self, synset):
		if synset.name() in self.synset_signature_cache:
			return self.synset_signature_cache[synset.name()]

		signature = set()
		for lemma in synset.lemma_names():
			signature.update(self.normalize_text(lemma.replace("_", " ")))
		signature.update(self.normalize_text(synset.definition()))
		for example in synset.examples():
			signature.update(self.normalize_text(example))
		for related_synset in synset.hypernyms() + synset.hyponyms():
			for lemma in related_synset.lemma_names():
				signature.update(self.normalize_text(lemma.replace("_", " ")))

		self.synset_signature_cache[synset.name()] = signature
		return signature

	def disambiguate_token(self, token, context_tokens):
		"""
		Choose the WordNet sense whose definition/example neighborhood overlaps
		most with the local document/query context. This is a lightweight Lesk
		style disambiguator; if context is uninformative, it falls back to the
		first WordNet sense.
		"""
		synsets = self.get_candidate_synsets(token)
		if not synsets:
			return None

		context = set()
		for context_token in context_tokens:
			normalized_context_token = self.normalize_token(context_token)
			normalized_token = self.normalize_token(token)
			if normalized_context_token and normalized_context_token != normalized_token:
				context.update(normalized_context_token.split())

		if not context:
			return synsets[0]

		best_synset = synsets[0]
		best_score = -1
		for candidate in synsets:
			signature = self.synset_signature(candidate)
			overlap = len(context.intersection(signature))
			lemma_bonus = 0
			for lemma in candidate.lemma_names():
				lemma_tokens = self.normalize_text(lemma.replace("_", " "))
				lemma_bonus += sum(1 for lemma_token in lemma_tokens if lemma_token in context)
			score = overlap + lemma_bonus
			if score > best_score:
				best_score = score
				best_synset = candidate

		return best_synset

	def token_to_concept_ids(self, token, concept_to_id, id_to_concept, allow_new=True, context_tokens=None):
		# Keep synsets as integer concept IDs internally instead of lemma/name strings.
		if context_tokens is None:
			context_tokens = []

		selected_synset = self.disambiguate_token(token, context_tokens)
		synsets = self.get_candidate_synsets(token)
		if selected_synset is not None:
			concept_keys = [selected_synset.name()]
		elif synsets:
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
		for idx, token in enumerate(tokens):
			left = max(0, idx - self.context_window_size)
			right = min(len(tokens), idx + self.context_window_size + 1)
			local_context = tokens[left:idx] + tokens[idx + 1:right]
			for concept_id in self.token_to_concept_ids(
				token,
				concept_to_id,
				id_to_concept,
				allow_new,
				local_context
			):
				concept_counts[concept_id] += 1
		return concept_counts

	def build_wsd_tfidf_index(self, docs, docIDs):
		total_docs = len(docIDs)
		doc_term_freq = {}
		doc_concept_freq = {}
		term_doc_freq = Counter()
		concept_doc_freq = Counter()
		doc_term_norms = {}
		doc_concept_norms = {}
		doc_order = {}
		concept_to_id = {}
		id_to_concept = []

		for position, docID in enumerate(docIDs):
			doc_order[docID] = position
			tokens = self.flatten_document(docs[position])

			term_counts = Counter(tokens)
			concept_counts = self.build_concept_counter(
				tokens,
				concept_to_id,
				id_to_concept,
				allow_new=True
			)

			doc_term_freq[docID] = term_counts
			doc_concept_freq[docID] = concept_counts

			for term in term_counts:
				term_doc_freq[term] += 1
			for concept_id in concept_counts:
				concept_doc_freq[concept_id] += 1

		term_idf = {}
		for term, df in term_doc_freq.items():
			term_idf[term] = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0

		concept_idf = {}
		for concept_id, df in concept_doc_freq.items():
			concept_idf[concept_id] = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0

		for docID in docIDs:
			term_norm_square = 0.0
			for term, tf in doc_term_freq[docID].items():
				weight = tf * term_idf[term]
				term_norm_square += weight * weight
			doc_term_norms[docID] = math.sqrt(term_norm_square)

			concept_norm_square = 0.0
			for concept_id, tf in doc_concept_freq[docID].items():
				weight = tf * concept_idf[concept_id]
				concept_norm_square += weight * weight
			doc_concept_norms[docID] = math.sqrt(concept_norm_square)

		return {
			"mode": "wsd_tfidf",
			"docIDs": list(docIDs),
			"doc_order": doc_order,
			"doc_term_freq": doc_term_freq,
			"doc_concept_freq": doc_concept_freq,
			"term_idf": term_idf,
			"concept_idf": concept_idf,
			"doc_term_norms": doc_term_norms,
			"doc_concept_norms": doc_concept_norms,
			"concept_to_id": concept_to_id,
			"id_to_concept": id_to_concept
		}

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

	def build_bm25_index(self, docs, docIDs):
		total_docs = len(docIDs)
		doc_term_freq = {}
		doc_freq = Counter()
		doc_lengths = {}
		doc_order = {}
		total_length = 0.0

		for position, docID in enumerate(docIDs):
			doc_order[docID] = position
			tokens = self.flatten_document(docs[position])
			term_counts = Counter(tokens)
			doc_term_freq[docID] = term_counts
			doc_lengths[docID] = len(tokens)
			total_length += len(tokens)

			for term in term_counts:
				doc_freq[term] += 1

		avg_doc_length = (total_length / total_docs) if total_docs > 0 else 0.0
		idf = {}
		for term, df in doc_freq.items():
			idf[term] = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))

		return {
			"mode": "bm25",
			"docIDs": list(docIDs),
			"doc_order": doc_order,
			"doc_term_freq": doc_term_freq,
			"doc_lengths": doc_lengths,
			"avg_doc_length": avg_doc_length,
			"idf": idf,
			"k1": 1.2,
			"b": 0.75
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
		k = min(150, max(1, int(math.sqrt(max_rank) * 8)))

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
		elif self.retrieval_mode == "bm25":
			self.index = self.build_bm25_index(docs, docIDs)
		elif self.retrieval_mode == "tfidf":
			self.index = self.build_tfidf_index(docs, docIDs)
		else:
			self.index = self.build_wsd_tfidf_index(docs, docIDs)

	def rank_bm25(self, queries):
		doc_IDs_ordered = []

		docIDs = self.index["docIDs"]
		doc_order = self.index["doc_order"]
		doc_term_freq = self.index["doc_term_freq"]
		doc_lengths = self.index["doc_lengths"]
		avg_doc_length = self.index["avg_doc_length"]
		idf = self.index["idf"]
		k1 = self.index["k1"]
		b = self.index["b"]

		for query in queries:
			query_tokens = self.flatten_document(query)
			query_tf = Counter(query_tokens)
			doc_scores = []

			for docID in docIDs:
				score = 0.0
				doc_tf = doc_term_freq[docID]
				doc_length = doc_lengths[docID]
				length_norm = 1.0 - b
				if avg_doc_length > 0.0:
					length_norm += b * (doc_length / avg_doc_length)

				for term, qtf in query_tf.items():
					if term not in idf or term not in doc_tf:
						continue
					tf = doc_tf[term]
					numerator = tf * (k1 + 1.0)
					denominator = tf + (k1 * length_norm)
					score += idf[term] * (numerator / denominator) * qtf

				doc_scores.append((score, doc_order[docID], docID))

			doc_scores.sort(key=lambda item: (-item[0], item[1]))
			doc_IDs_ordered.append([docID for _, _, docID in doc_scores])

		return doc_IDs_ordered

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

	def rank_wsd_tfidf(self, queries):
		doc_IDs_ordered = []

		docIDs = self.index["docIDs"]
		doc_order = self.index["doc_order"]
		doc_term_freq = self.index["doc_term_freq"]
		doc_concept_freq = self.index["doc_concept_freq"]
		term_idf = self.index["term_idf"]
		concept_idf = self.index["concept_idf"]
		doc_term_norms = self.index["doc_term_norms"]
		doc_concept_norms = self.index["doc_concept_norms"]
		concept_to_id = self.index["concept_to_id"]
		id_to_concept = self.index["id_to_concept"]

		for query in queries:
			query_tokens = self.flatten_document(query)
			query_tf = Counter(query_tokens)
			query_concepts = self.build_concept_counter(
				query_tokens,
				concept_to_id,
				id_to_concept,
				allow_new=False
			)

			query_term_weights = {}
			query_term_norm_square = 0.0
			for term, tf in query_tf.items():
				if term in term_idf:
					weight = tf * term_idf[term]
					query_term_weights[term] = weight
					query_term_norm_square += weight * weight

			query_concept_weights = {}
			query_concept_norm_square = 0.0
			for concept_id, tf in query_concepts.items():
				if concept_id in concept_idf:
					weight = tf * concept_idf[concept_id]
					query_concept_weights[concept_id] = weight
					query_concept_norm_square += weight * weight

			query_term_norm = math.sqrt(query_term_norm_square)
			query_concept_norm = math.sqrt(query_concept_norm_square)
			doc_scores = []

			for docID in docIDs:
				term_dot = 0.0
				doc_tf = doc_term_freq[docID]
				for term, query_weight in query_term_weights.items():
					if term in doc_tf:
						term_dot += query_weight * doc_tf[term] * term_idf[term]

				term_score = 0.0
				doc_term_norm = doc_term_norms[docID]
				if query_term_norm > 0.0 and doc_term_norm > 0.0:
					term_score = term_dot / (query_term_norm * doc_term_norm)

				concept_dot = 0.0
				doc_cf = doc_concept_freq[docID]
				for concept_id, query_weight in query_concept_weights.items():
					if concept_id in doc_cf:
						concept_dot += query_weight * doc_cf[concept_id] * concept_idf[concept_id]

				concept_score = 0.0
				doc_concept_norm = doc_concept_norms[docID]
				if query_concept_norm > 0.0 and doc_concept_norm > 0.0:
					concept_score = concept_dot / (query_concept_norm * doc_concept_norm)

				# Lexical evidence remains dominant; WSD concepts sharpen ambiguous matches.
				score = (0.75 * term_score) + (0.25 * concept_score)
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
		if self.index["mode"] == "bm25":
			return self.rank_bm25(queries)
		if self.index["mode"] == "wsd_tfidf":
			return self.rank_wsd_tfidf(queries)
		return self.rank_tfidf(queries)
