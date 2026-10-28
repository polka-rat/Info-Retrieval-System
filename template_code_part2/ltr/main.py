from sys import version_info
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from evaluation import Evaluation

import argparse
import json
import matplotlib.pyplot as plt
import time
import random
from importlib import import_module


# Input compatibility
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version")


class SearchEngine:

    def __init__(self, args):

        self.args = args

        if not os.path.exists(self.args.out_folder):
            os.makedirs(self.args.out_folder)

        self.tokenizer = Tokenization()

        self.sentenceSegmenter = SentenceSegmentation()

        self.inflectionReducer = InflectionReduction()

        self.stopwordRemover = StopwordRemoval()

        self.evaluator = Evaluation()

        self.ltr = self.createLTRRanker()

    def createLTRRanker(self):
        """
        Dynamically create LTR ranker based on configuration
        """
        ranker_configs = {
            "tfidf": {
                "module": "ltr.ltranker",
                "class": "LTRRanker",
            },
            "bm25": {
                "module": "ltr.ltranker_BM25",
                "class": "LTRRanker",
            },
        }

        config = ranker_configs[self.args.ltr_ranker]
        module = import_module(config["module"])
        ranker_class = getattr(module, config["class"])
        ranker = ranker_class()
        return ranker

    # ---------------------------------------------------
    # SENTENCE SEGMENTATION
    # ---------------------------------------------------
    def segmentSentences(self, text):

        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)

        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    # ---------------------------------------------------
    # TOKENIZATION
    # ---------------------------------------------------
    def tokenize(self, text):

        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)

        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    # ---------------------------------------------------
    # STEMMING / LEMMATIZATION
    # ---------------------------------------------------
    def reduceInflection(self, text):

        return self.inflectionReducer.reduce(text)

    # ---------------------------------------------------
    # STOPWORD REMOVAL
    # ---------------------------------------------------
    def removeStopwords(self, text):

        return self.stopwordRemover.fromList(text)

    # ---------------------------------------------------
    # QUERY PREPROCESSING
    # ---------------------------------------------------
    def preprocessQueries(self, queries):

        segmentedQueries = []

        for query in queries:

            segmentedQueries.append(
                self.segmentSentences(query)
            )

        tokenizedQueries = []

        for query in segmentedQueries:

            tokenizedQueries.append(
                self.tokenize(query)
            )

        reducedQueries = []

        for query in tokenizedQueries:

            reducedQueries.append(
                self.reduceInflection(query)
            )

        stopwordRemovedQueries = []

        for query in reducedQueries:

            stopwordRemovedQueries.append(
                self.removeStopwords(query)
            )

        return stopwordRemovedQueries

    # ---------------------------------------------------
    # DOCUMENT PREPROCESSING
    # ---------------------------------------------------
    def preprocessDocs(self, docs):

        segmentedDocs = []

        for doc in docs:

            segmentedDocs.append(
                self.segmentSentences(doc)
            )

        tokenizedDocs = []

        for doc in segmentedDocs:

            tokenizedDocs.append(
                self.tokenize(doc)
            )

        reducedDocs = []

        for doc in tokenizedDocs:

            reducedDocs.append(
                self.reduceInflection(doc)
            )

        stopwordRemovedDocs = []

        for doc in reducedDocs:

            stopwordRemovedDocs.append(
                self.removeStopwords(doc)
            )

        return stopwordRemovedDocs

    # ---------------------------------------------------
    # EVALUATE DATASET
    # ---------------------------------------------------
    def evaluateDataset(self):

        start_time = time.time()

        # -----------------------------------------------
        # LOAD QUERIES
        # -----------------------------------------------
        queries_json = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_queries.json"
                ),
                'r'
            )
        )

        query_ids = [
            item["query number"]
            for item in queries_json
        ]

        queries = [
            item["query"]
            for item in queries_json
        ]

        processedQueries = self.preprocessQueries(
            queries
        )

        # -----------------------------------------------
        # LOAD DOCUMENTS
        # -----------------------------------------------
        docs_json = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_docs.json"
                ),
                'r'
            )
        )

        doc_ids = [
            item["id"]
            for item in docs_json
        ]

        docs = [
            item["body"]
            for item in docs_json
        ]

        processedDocs = self.preprocessDocs(
            docs
        )

        # -----------------------------------------------
        # BUILD LTR STATS
        # -----------------------------------------------
        self.ltr.build_stats(
            processedDocs,
            doc_ids
        )

        # -----------------------------------------------
        # LOAD RELEVANCE JUDGMENTS
        # -----------------------------------------------
        qrels = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_qrels.json"
                ),
                'r'
            )
        )

        # -----------------------------------------------
        # TRAIN / TEST SPLIT
        # -----------------------------------------------
        indices = list(
            range(len(query_ids))
        )

        random.seed(42)

        random.shuffle(indices)

        split = int(0.7 * len(indices))

        train_idx = indices[:split]

        test_idx = indices[split:]

        train_queries = [
            processedQueries[i]
            for i in train_idx
        ]

        train_query_ids = [
            query_ids[i]
            for i in train_idx
        ]

        test_queries = [
            processedQueries[i]
            for i in test_idx
        ]

        test_query_ids = [
            query_ids[i]
            for i in test_idx
        ]

        # -----------------------------------------------
        # PREPARE TRAINING DATA
        # -----------------------------------------------
        X_train, y_train, group_train = (
            self.ltr.prepare_dataset(
                train_queries,
                train_query_ids,
                qrels
            )
        )

        # -----------------------------------------------
        # TRAIN MODEL
        # -----------------------------------------------
        self.ltr.train(
            X_train,
            y_train,
            group_train
        )

        # -----------------------------------------------
        # RANK TEST QUERIES
        # -----------------------------------------------
        doc_IDs_ordered = self.ltr.rank(
            test_queries
        )

        # -----------------------------------------------
        # EVALUATION
        # -----------------------------------------------
        precisions = []
        recalls = []
        fscores = []

        MAPs = []
        nDCGs = []
        MRRs = []

        for k in range(1, 11):

            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered,
                test_query_ids,
                qrels,
                k
            )

            recall = self.evaluator.meanRecall(
                doc_IDs_ordered,
                test_query_ids,
                qrels,
                k
            )

            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered,
                test_query_ids,
                qrels,
                k
            )

            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered,
                test_query_ids,
                qrels,
                k
            )

            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered,
                test_query_ids,
                qrels,
                k
            )

            MRR = self.evaluator.meanReciprocalRank(
                doc_IDs_ordered,
                test_query_ids,
                qrels,
                k
            )

            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)

            MAPs.append(MAP)
            nDCGs.append(nDCG)
            MRRs.append(MRR)

            print(
                f"Precision, Recall, F-score @ {k}: "
                f"{precision}, {recall}, {fscore}"
            )

            print(
                f"MAP, nDCG, MRR @ {k}: "
                f"{MAP}, {nDCG}, {MRR}"
            )

        # -----------------------------------------------
        # PLOT
        # -----------------------------------------------
        plt.plot(
            range(1, 11),
            precisions,
            label="Precision"
        )

        plt.plot(
            range(1, 11),
            recalls,
            label="Recall"
        )

        plt.plot(
            range(1, 11),
            fscores,
            label="F-Score"
        )

        plt.plot(
            range(1, 11),
            MAPs,
            label="MAP"
        )

        plt.plot(
            range(1, 11),
            nDCGs,
            label="nDCG"
        )

        plt.plot(
            range(1, 11),
            MRRs,
            label="MRR"
        )

        plt.legend()

        plt.title(
            "LTR Evaluation - Cranfield Dataset"
        )

        plt.xlabel("k")

        plt.savefig(
            os.path.join(
                args.out_folder,
                "eval_plot.png"
            )
        )

        end_time = time.time()

        runtime = end_time - start_time

        print("\nTotal Runtime:", runtime, "seconds")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='main.py'
    )

    parser.add_argument(
        '-dataset',
        default="cranfield/"
    )

    parser.add_argument(
        '-out_folder',
        default="output/"
    )

    parser.add_argument(
        '-segmenter',
        default="punkt"
    )

    parser.add_argument(
        '-tokenizer',
        default="ptb"
    )

    parser.add_argument(
        '-ltr_ranker',
        default="tfidf",
        choices=["tfidf", "bm25"]
    )

    args = parser.parse_args()

    searchEngine = SearchEngine(args)

    searchEngine.evaluateDataset()