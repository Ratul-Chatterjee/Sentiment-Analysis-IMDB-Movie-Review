from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from src.utils import load_data, prepare_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'IMDB Dataset.csv'
LSTM_DIR = ROOT / 'models' / 'lstm'
LSTM_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = load_data(str(DATA_PATH))
    X_train_raw, X_test_raw, y_train, y_test = prepare_split(df)

    max_words = 10000
    max_len = 200
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_raw)
    X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
    X_test_seq = tokenizer.texts_to_sequences(X_test_raw)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    model = Sequential([
        Embedding(input_dim=max_words, output_dim=100, input_length=max_len),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(X_train_pad, y_train, validation_split=0.1, epochs=4, batch_size=128, callbacks=[es])

    loss, acc = model.evaluate(X_test_pad, y_test, verbose=1)
    print(f'LSTM test accuracy: {acc:.4f}')

    import json
    with open(LSTM_DIR / 'tokenizer.json', 'w') as f:
        f.write(tokenizer.to_json())
    model.save(LSTM_DIR / 'lstm_model.keras')

    results_path = ROOT / 'results' / 'reports.txt'
    with open(results_path, 'a') as f:
        f.write(f"\nLSTM test accuracy: {acc:.4f}\n")

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
