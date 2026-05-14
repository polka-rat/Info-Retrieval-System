from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval

from evaluation import Evaluation
import matplotlib.pyplot as plt
import time
import re
from sys import version_info
import argparse
import json
import os

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
        self.rank_till = 10
        if not os.path.exists(self.args.out_folder):
            os.makedirs(self.args.out_folder)

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

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

    def normalizeTokens(self, text):
        normalizedText = []
        for sent in text:
            normalizedSentence = []
            for token in sent:
                if not token:
                    continue

                token = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', token)
                if not token:
                    continue

                for part in token.split('-'):
                    part = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', part)
                    if part:
                        normalizedSentence.append(part)

            normalizedText.append(normalizedSentence)

        return normalizedText

    def preprocessQueries(self, queries):
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)

        json.dump(segmentedQueries, open(os.path.join(self.args.out_folder, "segmented_queries.txt"), 'w'))

        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)

        json.dump(tokenizedQueries, open(os.path.join(self.args.out_folder, "tokenized_queries.txt"), 'w'))

        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)

        json.dump(reducedQueries, open(os.path.join(self.args.out_folder, "reduced_queries.txt"), 'w'))

        normalizedQueries = []
        for query in reducedQueries:
            normalizedQuery = self.normalizeTokens(query)
            normalizedQueries.append(normalizedQuery)

        stopwordRemovedQueries = []
        for query in normalizedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)

        json.dump(stopwordRemovedQueries, open(os.path.join(self.args.out_folder, "stopword_removed_queries.txt"), 'w'))

        return stopwordRemovedQueries

    def preprocessDocs(self, docs):
        cache_path = os.path.join(self.args.out_folder, "pre_processed_docs.json")

        if os.path.exists(cache_path):
            print("Loading preprocessed documents from cache...")
            with open(cache_path, "r") as f:
                cachedDocs = json.load(f)

            normalizedDocs = [self.normalizeTokens(doc) for doc in cachedDocs]
            stopwordRemovedDocs = [self.removeStopwords(doc) for doc in normalizedDocs]

            with open(cache_path, "w") as f:
                json.dump(stopwordRemovedDocs, f)

            return stopwordRemovedDocs

        print("No cache found. Running preprocessing pipeline...")

        start = time.time()

        segmentedDocs = [self.segmentSentences(doc) for doc in docs]
        json.dump(segmentedDocs, open(os.path.join(self.args.out_folder, "segmented_docs.txt"), 'w'))

        tokenizedDocs = [self.tokenize(doc) for doc in segmentedDocs]
        json.dump(tokenizedDocs, open(os.path.join(self.args.out_folder, "tokenized_docs.txt"), 'w'))

        reducedDocs = [self.reduceInflection(doc) for doc in tokenizedDocs]
        json.dump(reducedDocs, open(os.path.join(self.args.out_folder, "reduced_docs.txt"), 'w'))

        normalizedDocs = [self.normalizeTokens(doc) for doc in reducedDocs]

        stopwordRemovedDocs = [self.removeStopwords(doc) for doc in normalizedDocs]
        json.dump(stopwordRemovedDocs, open(os.path.join(self.args.out_folder, "stopword_removed_docs.txt"), 'w'))

        # ---- 3. Save final cache ----
        with open(cache_path, "w") as f:
            json.dump(stopwordRemovedDocs, f)

        end = time.time()
        print(f"Preprocessing time for documents: {end - start:.2f} seconds")

        return stopwordRemovedDocs

    def evaluateDataset(self):

        queries_json = json.load(open(os.path.join(args.dataset, "cran_queries.json"), 'r'))[:]
        query_ids = [item["query number"] for item in queries_json]
        queries = [item["query"] for item in queries_json]

        processedQueries = self.preprocessQueries(queries)

        docs_json = json.load(open(os.path.join(args.dataset, "cran_docs.json"), 'r'))[:]
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processedDocs = self.preprocessDocs(docs)

        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

        qrels = json.load(open(os.path.join(args.dataset, "cran_qrels.json"), 'r'))[:]

        precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []
         
        for k in range(1, self.rank_till + 1):

            precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)

            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)

            print(f"Precision, Recall, F-score @ {k}: {precision}, {recall}, {fscore}")

            MAP = self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            MRR = self.evaluator.meanReciprocalRank(doc_IDs_ordered, query_ids, qrels, k)

            MAPs.append(MAP)
            nDCGs.append(nDCG)
            MRRs.append(MRR)

            print(f"MAP, nDCG, MRR @ {k}: {MAP}, {nDCG}, {MRR}")

        # Plot
        plt.plot(range(1, self.rank_till + 1), precisions, label="Precision")
        plt.plot(range(1, self.rank_till + 1), recalls, label="Recall")
        plt.plot(range(1, self.rank_till + 1), fscores, label="F-Score")
        plt.plot(range(1, self.rank_till + 1), MAPs, label="MAP")
        plt.plot(range(1, self.rank_till + 1), nDCGs, label="nDCG")
        plt.plot(range(1, self.rank_till + 1), MRRs, label="MRR")

        plt.legend()
        method_name = args.retriever.replace("_", " ").title()
        plt.title(f"Evaluation Metrics - Cranfield Dataset ({method_name})")
        plt.xlabel("k")
        plt.savefig(os.path.join(args.out_folder, "eval_plot.png"))

    def handleCustomQuery(self):

        print("Enter query below")
        query = input()

        processedQuery = self.preprocessQueries([query])[0]

        docs_json = json.load(open(os.path.join(args.dataset, "cran_docs.json"), 'r'))[:]
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processedDocs = self.preprocessDocs(docs)

        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('-dataset', default=r"C:\Users\Nagasai\OneDrive\Desktop\Acads\Sem 6\CS6370\NLP_Project\NLP Assignemnet\cranfield")
    parser.add_argument('-out_folder', default="output/")
    parser.add_argument('-segmenter', default="punkt")
    parser.add_argument('-tokenizer', default="ptb")
    parser.add_argument('-custom', action="store_true")

    args = parser.parse_args()

    searchEngine = SearchEngine(args)

    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
