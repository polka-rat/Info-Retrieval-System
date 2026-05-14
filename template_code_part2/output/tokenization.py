from util import *
from nltk.tokenize import TreebankWordTokenizer
import spacy

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

		tokenizedText = []
		# Treat these punctuation marks as standalone tokens in the naive tokenizer.
		separators = [".", "!", "?", ",", ";", ":", "(", ")", "\"", "'", "+", "=", "-"]

		for sent in text:
			tokens = []
			i = 0
			current = ""

			while i < len(sent):
				ch = sent[i]

				# Flush the current token whenever whitespace is encountered.
				if ch.isspace():
					if current:
						tokens.append(current)
						current = ""
				# Split punctuation into separate tokens using a fixed rule list.
				elif ch in separators:
					if current:
						tokens.append(current)
						current = ""
					tokens.append(ch)
				else:
					current += ch
				i += 1

			if current:
				tokens.append(current)
			# Drop any accidental whitespace-only tokens before storing the sentence.
			tokens = [tok for tok in tokens if tok.strip()]
			tokenizedText.append(tokens)
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
		tokenizedText = []
		# TreebankWordTokenizer applies Penn Treebank tokenization rules.
		tokenizer = TreebankWordTokenizer()

		for sent in text:
			# Keep only non-empty, non-whitespace tokens from the Treebank tokenizer.
			tokens = [tok for tok in tokenizer.tokenize(sent) if tok.strip()]
			tokenizedText.append(tokens)
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
		# Load spaCy's English pipeline and tokenize each sentence with its built-in tokenizer.
		nlp = spacy.load("en_core_web_sm")
		tokenizedText = []
		for sent in text:
			doc = nlp(sent)
			# Exclude spaCy space tokens so the output contains only visible tokens.
			tokens = [token.text for token in doc if token.text.strip()]
			tokenizedText.append(tokens)
		return tokenizedText
