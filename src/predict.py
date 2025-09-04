from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from src.utils import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models' / 'sklearn_models'
LSTM_DIR = ROOT / 'models' / 'lstm'

def predict_classical(texts, model_name='logistic_regression'):
    vect = joblib.load(MODELS_DIR / 'tfidf_vectorizer.joblib')
    model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    clean_texts = [clean_text(t) for t in texts]
    X = vect.transform(clean_texts)
    preds = model.predict(X)
    return ['positive' if p == 1 else 'negative' for p in preds]

def predict_lstm(texts):
    import json
    with open(LSTM_DIR / 'tokenizer.json') as f:
        tok_json = f.read()
    tokenizer = tokenizer_from_json(tok_json)
    seq = tokenizer.texts_to_sequences([clean_text(t) for t in texts])
    pad = pad_sequences(seq, maxlen=200)
    model = load_model(LSTM_DIR / 'lstm_model.keras')
    probs = model.predict(pad)
    return ['positive' if p >= 0.5 else 'negative' for p in probs.flatten()]

if __name__ == '__main__':
    sample = ["I loved the movie, it was a thrilling ride!", "I hated this movie. It was so boring."]
    print('Logistic preds:', predict_classical(sample, 'logistic_regression'))
    print('LSTM preds:', predict_lstm(sample))
