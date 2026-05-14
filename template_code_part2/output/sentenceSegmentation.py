from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import PunktTokenizer


class SentenceSegmentation():

	def __init__(self):
		# Load spaCy lazily so punkt/naive modes do not require the model.
		self.nlp = None

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		# End a sentence whenever one of the sentence-final punctuation marks appears.
		separators = [".", "!", "?"]

		sentences = []
		i = 0
		current = ""

		while i < len(text):
			ch = text[i]
			if ch in separators:
				# Include the separator in the current sentence before storing it.
				current += ch
				sentences.append(current.strip())
				current = ""

			else:
				current += ch

			i += 1

		if current:
			sentences.append(current)

		# Fill in code here
		return sentences


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# Punkt uses a learned statistical model instead of only punctuation rules.
		tokenizer = PunktTokenizer()
		segmentedText = tokenizer.tokenize(text.strip())

		# Fill in code here

		return segmentedText


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		if self.nlp is None:
			# Only require the spaCy model when the spaCy segmenter is used.
			self.nlp = spacy.load("en_core_web_sm")
		# spaCy exposes sentence spans through the Doc.sents iterator.
		sents = self.nlp(text).sents
		segmentedText = [sent.text for sent in sents]

		return segmentedText
