"""
src package for sentiment-analysis-imdb project.

Modules:
- utils.py : Data loading, cleaning, splitting.
- train_classical.py : Train Logistic Regression, Naive Bayes, SVM, Random Forest.
- train_lstm.py : Train an LSTM model with Keras.
- predict.py : Load saved models and make predictions.
"""

__all__ = ["utils", "train_classical", "train_lstm", "predict"]
