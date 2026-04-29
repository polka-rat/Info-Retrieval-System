from util import *
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)


class InflectionReduction:

	def _get_wordnet_pos(self, word):
		tag = nltk.pos_tag([word])[0][1][0].upper()
		tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
		return tag_map.get(tag, wordnet.NOUN)

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
		porter = PorterStemmer()
		reducedText = None
		reducedText = [[porter.stem(s) for s in sentence] for sentence in text]
		# Fill in code here

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
		wnl = WordNetLemmatizer()
		reducedText = None
		reducedText = [[wnl.lemmatize(s, self._get_wordnet_pos(s)) for s in sentence] for sentence in text]
		# Fill in code here

		return reducedText



	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		reducedText = None
		reducedText = self.wordnetLemmatizer(text)
		# Fill in code here

		return reducedText
