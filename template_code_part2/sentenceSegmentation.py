from util import *

# Add your import statements here
import re
import nltk
from nltk.tokenize import sent_tokenize

try:
	import spacy
except Exception:
	spacy = None


class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model if available in the current environment.
		self.nlp = None
		if spacy is not None:
			try:
				self.nlp = spacy.load("en_core_web_sm")
			except Exception:
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

		segmentedText = [s.strip() for s in text.split('.') if s.strip()]
  

		return segmentedText


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

		segmentedText = [s.strip() for s in sent_tokenize(text) if s.strip()]

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

		# Fall back to Punkt if spaCy/model is unavailable.
		if self.nlp is None:
			return self.punkt(text)

		segmentedText = [s.text.strip() for s in list(self.nlp(text).sents) if s.text.strip()]

		# Fill in code here

		return segmentedText
