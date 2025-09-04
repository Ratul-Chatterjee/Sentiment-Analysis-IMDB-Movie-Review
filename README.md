# Sentiment-Analysis-IMDB-Movie-Review

Deep Learning project for Sentiment Analysis on the IMDB Movie Reviews dataset. This repository trains and evaluates four classical ML models and an LSTM deep learning model to classify movie reviews as **positive** or **negative**.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Dataset](#dataset)
* [Requirements](#requirements)
* [Quick Start](#quick-start)
* [Usage Examples](#usage-examples)
* [Modeling Details](#modeling-details)
* [Results & Output](#results--output)
  
---

## Project Overview

This project demonstrates end-to-end sentiment analysis on the IMDB movie reviews dataset using both classical machine learning algorithms (e.g., Logistic Regression, Naive Bayes, SVM, Random Forest — four classical models) and a recurrent neural network (LSTM). The goal is to compare classical NLP pipelines (tokenization → TF-IDF/Count → classical classifier) with a deep learning approach (tokenization → embedding → LSTM).

## Features

* Preprocessing pipeline for text cleaning, tokenization and sequence preparation
* TF-IDF and classical model training and evaluation
* Embedding + LSTM model training and evaluation
* Model saving/loading (under `models/`)
* Results and visualizations saved under `results/`

## Repository Structure

```
Sentiment-Analysis-IMDB-Movie-Review/
├── data/               # raw or processed datasets (IMDB train/test or csv files)
├── models/             # saved model files (pickle / .h5)
├── results/            # evaluation metrics, plots, logs
├── src/                # source code: preprocessing, training, evaluation scripts
│   ├── preprocess.py   # text cleaning, tokenization helpers
│   ├── train_classical.py # train classical ML models (TF-IDF + classifier)
│   ├── train_lstm.py   # build and train LSTM model
│   ├── evaluate.py     # evaluation utilities and plotting
│   └── utils.py        # helper functions
├── README.md           # this file
└── requirements.txt    # Python dependencies
```

## Dataset

This project uses the IMDB Movie Review dataset (binary sentiment labels). You can either:

1. Use the prepackaged dataset from `keras.datasets.imdb` or `tensorflow.keras.datasets`.
2. Download the popular IMDB dataset (50k labeled reviews) from Kaggle.
   
Place raw or processed files under `data/` as dataset.

## Requirements

Create a virtual environment and install dependencies.

```bash
python -m venv venv
source venv/bin/activate    # on macOS / Linux
venv\Scripts\activate     # on Windows
pip install -r requirements.txt
```

A suggested `requirements.txt` (adjust versions as needed):

```
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow   # or tensorflow-cpu if you don't have a GPU
keras
nltk
tqdm
joblib
```

## Quick Start

1. Prepare data

* If you already have the IMDB dataset files, place them into `data/`.
* If using the built-in keras dataset, scripts will typically download it automatically.

2. Preprocess (example):

```bash
python src/preprocess.py --input data/raw_imdb.csv --output data/processed
```

3. Train classical models (TF-IDF + classifiers):

```bash
python src/train_classical.py --data data/processed --out models/classical/
```

4. Train LSTM model:

```bash
python src/train_lstm.py --data data/processed --out models/lstm/
```

5. Evaluate and view results:

```bash
python src/evaluate.py --models models/ --results results/
```

## Usage Examples

Below are *example* command-line usages. Replace script names/arguments to match the ones available in `src/`.

* Train a Logistic Regression with TF-IDF:

```bash
python src/train_classical.py --model logistic --tfidf --max-features 20000
```

* Train an LSTM (example hyperparameters):

```bash
python src/train_lstm.py --embedding-dim 100 --maxlen 300 --batch-size 64 --epochs 10
```

* Evaluate saved models and produce ROC/confusion matrix plots:

```bash
python src/evaluate.py --model-dir models/ --save-dir results/
```

## Modeling Details

**Classical workflow**

* Text cleaning: lowercasing, removing HTML tags, punctuation, stopwords (optional), and tokenization.
* Vectorization: CountVectorizer / TF-IDF with configurable `ngram_range` and `max_features`.
* Classifiers: Logistic Regression, Multinomial Naive Bayes, Support Vector Machine (LinearSVC), Random Forest (typical four classical choices).

**Deep learning workflow (LSTM)**

* Tokenize text and convert to integer sequences.
* Pad sequences to fixed `maxlen`.
* Embedding layer (trainable or with pre-trained embeddings like GloVe if available).
* One or more LSTM layers, optional dropout, followed by dense layer with sigmoid for binary output.

## Results & Output

All experiment artifacts (trained models, evaluation metrics, and plots) should be saved under the `models/` and `results/` folders respectively. Typical outputs:

* `models/lstm/model.h5` — trained LSTM weights
* `models/classical/logistic.pkl` — pickled sklearn model
* `results/metrics.csv` — per-model accuracy, precision, recall, F1
* `results/confusion_matrix_{model}.png` — confusion matrices
