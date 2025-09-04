import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.utils import load_data, prepare_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'IMDB Dataset.csv'
MODELS_DIR = ROOT / 'models' / 'sklearn_models'
RESULTS = ROOT / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

def main():
    df = load_data(str(DATA_PATH))
    X_train, X_test, y_train, y_test = prepare_split(df)

    vect = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'naive_bayes': MultinomialNB(),
        'linear_svm': LinearSVC(max_iter=2000),
        'random_forest': RandomForestClassifier(n_estimators=100, n_jobs=-1)
    }

    report_lines = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc:.4f}")
        report_lines.append(f"Model: {name}\nAccuracy: {acc:.4f}\n")
        report_lines.append(classification_report(y_test, preds, digits=4))

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    joblib.dump(vect, MODELS_DIR / 'tfidf_vectorizer.joblib')

    with open(RESULTS / 'reports.txt', 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == '__main__':
    main()
