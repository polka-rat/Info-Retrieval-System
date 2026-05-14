from util import *

# Add your import statements here
from nltk.corpus import stopwords
import nltk



class StopwordRemoval():
	def __init__(self):
		# Ensure the NLTK stopword list is present before removal is attempted.
		nltk.download('stopwords')
	
	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		# Use the curated English stopword list provided by NLTK.
		stop_words = set(stopwords.words('english'))
		stop_words.update({
			".", ",", "!", "?", ";", ":", "'", "\"", "(", ")", "[", "]",
			"{", "}", "-", "_", "/", "\\", "@", "#", "$", "%", "^", "&",
			"*", "+", "=", "<", ">", "|", "~", "`", "``", "''", "--", "...", "..", "'s", ",",
			"-",".","a","an","and","are", "at", "by", "flow", "for", "in","is","of","on","that",
			"the","to","with"
		})
		stopwordRemovedText = []
		for sent in text:
			# Preserve sentence boundaries while filtering out stopwords token by token.
			filtered_tokens = [w for w in sent if not w.lower() in stop_words]
			stopwordRemovedText.append(filtered_tokens)
		return stopwordRemovedText

	



	
