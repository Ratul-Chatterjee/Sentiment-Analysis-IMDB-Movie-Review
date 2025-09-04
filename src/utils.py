import pandas as pd
from sklearn.model_selection import train_test_split
import re

def load_data(path: str = 'data/IMDB Dataset.csv'):
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    return df

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", ' ', text)
    text = re.sub(r"[^a-z0-9\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def prepare_split(df, test_size=0.2, random_state=42):
    df['review'] = df['review'].astype(str).apply(clean_text)
    X = df['review'].values
    y = (df['sentiment'] == 'positive').astype(int).values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
