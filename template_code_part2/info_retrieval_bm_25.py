import math
import re
from collections import Counter

import numpy as np
from nltk.corpus import wordnet as wn


class InformationRetrieval():

	def __init__(self, retrieval_mode="bm25"):
		self.index = None
		self.retrieval_mode = retrieval_mode
		self.synset_signature_cache = {}
		self.synset_lookup_cache = {}
		self.context_window_size = 4

	def setRetrievalMode(self, retrieval_mode):
		"""
		Switch between supported retrieval strategies.

		Supported values:
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

	def buildIndex(self, docs, docIDs):
		"""
		Build the BM25 index for the provided documents.
		"""
		self.index = self.build_bm25_index(docs, docIDs)

	def rank(self, queries):
		"""
		Rank documents using BM25.
		"""
		if self.index is None:
			return []
		return self.rank_bm25(queries)
