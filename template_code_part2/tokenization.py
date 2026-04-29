from util import *
from nltk.tokenize.treebank import TreebankWordTokenizer
# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None
		tokenizedText = [s.split() for s in text]
		tokenizedText = [[t.strip().lower() for t in sentence if t.strip()] for sentence in tokenizedText]
  
		# Fill in code here

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None
		tokenizer=TreebankWordTokenizer()
		tokenizedText = [tokenizer.tokenize(s) for s in text]
		# Fill in code here

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None
		tokenizedText = [[token.text for token in self.nlp(s)] for s in text]
		# Fill in code here

		return tokenizedText
