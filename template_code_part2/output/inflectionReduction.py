from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
from nltk.stem import PorterStemmer, WordNetLemmatizer
import copy
import nltk

class InflectionReduction:
	def __init__(self):
		# Ensure the WordNet resource is available before lemmatization is used.
		nltk.download('wordnet')
	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		# Work on a copy so the original tokenized text is not modified in place.
		reducedText = copy.deepcopy(text)

		# Fill in code here
		porter = PorterStemmer()
		for sent in reducedText:
			for i in range(len(sent)):
				sent[i] = porter.stem(sent[i])

		return reducedText



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		# Work on a copy so stemming and lemmatization can be compared independently.
		reducedText = copy.deepcopy(text)

		# Fill in code here
		
		# Lemmatize each token sentence by sentence using WordNet.
		lemmatizer = WordNetLemmatizer()
		for sent in reducedText:
			for i in range(len(sent)):
				sent[i] = lemmatizer.lemmatize(sent[i])

		return reducedText




	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		reducedText = None

		# Fill in code here

		# Default to the lemmatizer for the assignment pipeline.
		return self.wordnetLemmatizer(text)
