## README (Part 2: Search Engine Implementation)

This folder contains the additional files required for Part 2 of the assignment, involving the construction of a basic Information Retrieval (IR) system using the Vector Space Model.

The code is compatible with both Python 2 and Python 3.

---

###  Folder Structure

Ensure that your project directory is organized as follows:

```
project_folder/
│
├── main.py
├── informationRetrieval.py
├── evaluation.py
├── sentenceSegmentation.py
├── tokenization.py
├── inflectionReduction.py
├── stopwordRemoval.py
├── util.py
│
├── cranfield/
│   ├── cran_docs.json
│   ├── cran_queries.json
│   ├── cran_qrels.json
│
└── output/
```

* The `cranfield/` directory must contain the dataset files.
* The `output/` directory will be used to store intermediate preprocessing outputs and evaluation plots.

---

### Important Instructions

* Implement the required functions in:

  * `informationRetrieval.py`
  * `evaluation.py`

* **Do NOT modify `main.py`.**

* You may use any Python libraries such as `nltk`, `math`, etc.

---

###  Requirements

Install the required dependencies:

```
pip install nltk matplotlib
```

Run the following once in Python to download necessary resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

### ▶ Running the Code

####  1. Run on Full Dataset (Evaluation Mode)

```
python main.py -dataset cranfield/ -out_folder output/

```

This will:

* Process all queries in the dataset
* Compute evaluation metrics:

  * Precision@k
  * Recall@k
  * F-score@k
  * nDCG@k
  * Mean Average Precision (MAP)
* Generate output files in the `output/` folder
* Save evaluation plots as `eval_plot.png`

---
To run custom retrievers:
```
python main.py -retriever ${retriever} -dataset cranfield -out_folder output
```

Available retrievers:
* **bm25** - BM25 ranking algorithm (probabilistic relevance framework)
* **bm25_plus** - BM25+ variant with delta parameter for improved ranking
* **bm25_reranker** - Two-stage retrieval: BM25+ with neural reranking using CrossEncoder
* **query_expansion** - Query Expansion (QE-LSA): TF-IDF + LSA-based term expansion for richer queries
* **lsa** - Latent Semantic Analysis with dimensionality reduction
* **ngram** - N-gram based token sequence matching
* **lsa_synset** - LSA combined with WordNet synsets for semantic understanding
* **sdmlsa_hybrid** - Hybrid approach combining multiple retrieval methods
* **sentence_wise** - Sentence-wise retrieval using Word2Vec embeddings and document centroids

Examples:
```
python main.py -retriever bm25_plus -dataset cranfield -out_folder output
python main.py -retriever bm25_reranker -dataset cranfield -out_folder output
python main.py -retriever query_expansion -dataset cranfield -out_folder output
```

####  2. Run in Custom Query Mode

```
python main.py -custom -dataset cranfield/ -out_folder output/
```

You will be prompted to enter a query:

```
Enter query below
Papers on Aerodynamics
```

The system will output the IDs of the top 5 most relevant documents.

---

###  Output Files

After execution, the following files will be generated in the `output/` directory:

* `segmented_queries.txt`
* `tokenized_queries.txt`
* `reduced_queries.txt`
* `stopword_removed_queries.txt`
* `segmented_docs.txt`
* `tokenized_docs.txt`
* `reduced_docs.txt`
* `stopword_removed_docs.txt`
* `eval_plot.png`

These files correspond to intermediate preprocessing stages and final evaluation results.

---

###  Notes

* Ensure the `cranfield/` dataset folder path is correct.
* Ensure the `output/` folder exists (or create it manually if needed).
* If evaluation metrics return incorrect values (e.g., -1), verify your implementations in `evaluation.py`.

---

### 🎯 Summary

* Do NOT modify `main.py`
* Implement required logic in:

  * `informationRetrieval.py`
  * `evaluation.py`
* Run using the commands provided above

---
