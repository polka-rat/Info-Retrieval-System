import math
import re
from collections import Counter

import numpy as np
from nltk.corpus import wordnet as wn


class InformationRetrieval():

	def __init__(self, k=100):
		self.index = None
		self.k = k
		self.synset_signature_cache = {}
		self.synset_lookup_cache = {}
		self.context_window_size = 4

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
		self.index = self.build_lsa_synset_index(docs, docIDs)

	def rank(self, queries):
		if self.index is None:
			return []

		docIDs = self.index["docIDs"]
		doc_order = self.index["doc_order"]
		concept_to_id = self.index["concept_to_id"]
		id_to_concept = self.index["id_to_concept"]
		concept_to_row = self.index["concept_to_row"]
		idf = self.index["idf"]
		uk = self.index["uk"]
		sk = self.index["sk"]
		doc_vectors = self.index["doc_vectors"]

		doc_IDs_ordered = []
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
