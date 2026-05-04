from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval

import argparse
import json
import os
from sys import version_info


# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		# Create output folder if not exists
		if not os.path.exists(self.args.out_folder):
			os.makedirs(self.args.out_folder)

	def segmentSentences(self, text):
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):

		segmentedQueries = []
		for query in queries:
			segmentedQueries.append(self.segmentSentences(query))

		with open(os.path.join(self.args.out_folder, "segmented_queries.txt"), 'w') as f:
			json.dump(segmentedQueries, f)

		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQueries.append(self.tokenize(query))

		with open(os.path.join(self.args.out_folder, "tokenized_queries.txt"), 'w') as f:
			json.dump(tokenizedQueries, f)

		reducedQueries = []
		for query in tokenizedQueries:
			reducedQueries.append(self.reduceInflection(query))

		with open(os.path.join(self.args.out_folder, "reduced_queries.txt"), 'w') as f:
			json.dump(reducedQueries, f)

		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQueries.append(self.removeStopwords(query))

		with open(os.path.join(self.args.out_folder, "stopword_removed_queries.txt"), 'w') as f:
			json.dump(stopwordRemovedQueries, f)

		return stopwordRemovedQueries


	def preprocessDocs(self, docs):

		segmentedDocs = []
		for doc in docs:
			segmentedDocs.append(self.segmentSentences(doc))

		with open(os.path.join(self.args.out_folder, "segmented_docs.txt"), 'w') as f:
			json.dump(segmentedDocs, f)

		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDocs.append(self.tokenize(doc))

		with open(os.path.join(self.args.out_folder, "tokenized_docs.txt"), 'w') as f:
			json.dump(tokenizedDocs, f)

		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDocs.append(self.reduceInflection(doc))

		with open(os.path.join(self.args.out_folder, "reduced_docs.txt"), 'w') as f:
			json.dump(reducedDocs, f)

		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDocs.append(self.removeStopwords(doc))

		with open(os.path.join(self.args.out_folder, "stopword_removed_docs.txt"), 'w') as f:
			json.dump(stopwordRemovedDocs, f)

		return stopwordRemovedDocs


	def evaluateDataset(self):

		query_path = os.path.join(self.args.dataset, "cran_queries.json")
		doc_path = os.path.join(self.args.dataset, "cran_docs.json")

		with open(query_path, 'r') as f:
			queries_json = json.load(f)

		queries = [item["query"] for item in queries_json]
		self.preprocessQueries(queries)

		with open(doc_path, 'r') as f:
			docs_json = json.load(f)

		docs = [item["body"] for item in docs_json]
		self.preprocessDocs(docs)


	def handleCustomQuery(self):

		print("Enter query below")
		query = input()

		self.preprocessQueries([query])

		doc_path = os.path.join(self.args.dataset, "cran_docs.json")

		with open(doc_path, 'r') as f:
			docs_json = json.load(f)

		docs = [item["body"] for item in docs_json]
		self.preprocessDocs(docs)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='main.py')

	parser.add_argument('-dataset', default="cranfield",
						help="Path to dataset folder")

	parser.add_argument('-out_folder', default="output",
						help="Path to output folder")

	parser.add_argument('-segmenter', default="punkt",
	                    help="Sentence Segmenter Type [naive|punkt]")

	parser.add_argument('-tokenizer', default="ptb",
	                    help="Tokenizer Type [naive|ptb]")

	parser.add_argument('-custom', action="store_true",
						help="Take custom query as input")

	args = parser.parse_args()

	searchEngine = SearchEngine(args)

	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
