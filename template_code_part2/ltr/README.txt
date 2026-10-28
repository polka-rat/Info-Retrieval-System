## README (LTR: Learning-to-Rank)

This folder contains Learning-to-Rank (LTR) implementation using LightGBM for document ranking on the Cranfield dataset.

---

### Folder Structure

```
ltr/
├── main.py
├── ltranker.py
├── ltranker_BM25.py
└── README.txt
```

* `main.py` - Main pipeline for preprocessing, training, and evaluation
* `ltranker.py` - TF-IDF based feature extraction for LTR
* `ltranker_BM25.py` - BM25-based feature extraction for LTR

---

### Requirements

Install the required dependencies:

```
pip install nltk matplotlib lightgbm numpy
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

#### Run LTR with TF-IDF Features (Default)

```
python ltr/main.py -dataset cranfield/ -out_folder output/
```

This will:
* Use TF-IDF based feature extraction (ltranker.py)
* Split dataset into 70% training and 30% test
* Train LightGBM model on training queries
* Evaluate on test queries
* Compute evaluation metrics (Precision@k, Recall@k, F-score@k, MAP, nDCG, MRR)
* Generate evaluation plot as `eval_plot.png`

#### Run LTR with BM25 Features

```
python ltr/main.py -ltr_ranker bm25 -dataset cranfield/ -out_folder output/
```

This will:
* Use BM25-based feature extraction (ltranker_BM25.py)
* Same pipeline as TF-IDF variant
* Different ranking signals through BM25 scoring

---

### Available LTR Rankers

* **tfidf** (default) - TF-IDF based feature extraction with 3 features:
  - TF-IDF score
  - Query term frequency
  - Document length normalization

* **bm25** - BM25-based feature extraction with 5 features:
  - BM25 score (main ranking signal)
  - Query term coverage (recall proxy)
  - Term frequency match strength
  - Document length penalty
  - Query frequency importance signal

---

### Command-line Arguments

```
-dataset DATASET          Path to dataset folder (default: cranfield/)
-out_folder OUT_FOLDER    Output folder path (default: output/)
-segmenter SEGMENTER      Sentence segmenter: 'naive' or 'punkt' (default: punkt)
-tokenizer TOKENIZER      Tokenizer: 'naive' or 'ptb' (default: ptb)
-ltr_ranker {tfidf,bm25}  LTR ranker type (default: tfidf)
```

---

### Example Usage

```bash
# Use BM25 ranker with custom dataset path
python ltr/main.py -ltr_ranker bm25 -dataset cranfield/ -out_folder results/

# Use TF-IDF ranker (default)
python ltr/main.py -dataset cranfield/ -out_folder results/

# Use with different segmenter and tokenizer
python ltr/main.py -ltr_ranker bm25 -segmenter naive -tokenizer naive -dataset cranfield/
```

---

### Output Files

The evaluation will generate the following in the output folder:

* `eval_plot.png` - Plot showing Precision, Recall, F-score, MAP, nDCG, and MRR curves
* Console output with detailed metrics @ k=1 to k=10

---

### Training/Test Split

* **Default split**: 70% training, 30% test
* **Random seed**: 42 (for reproducibility)
* **Feature extraction**: Both rankers extract features for all (query, document) pairs

---

### Notes

- Both rankers use the same training pipeline and evaluation metrics
- The main difference is in the feature extraction method (TF-IDF vs BM25)
- LightGBM model is trained on the extracted features with relevance labels
