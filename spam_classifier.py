import re
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


SPAM_WORDS = {
    "free","win","winner","prize","urgent","offer","limited","money","cash",
    "click","buy","cheap","discount","deal","bonus","viagra","crypto","investment",
    "guaranteed","congratulations","act","now"
}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def extract_features(email_text: str) -> dict:
    # words
    tokens = re.findall(r"[A-Za-z']+", email_text)
    words = len(tokens)

    # links
    links = len(URL_RE.findall(email_text))

    # capital words (ALL CAPS tokens length>=2)
    capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())

    # spam_word_count (count spammy tokens)
    spam_word_count = sum(1 for t in tokens if t.lower() in SPAM_WORDS)

    return {
        "words": words,
        "links": links,
        "capital_words": capital_words,
        "spam_word_count": spam_word_count
    }


def train_model(csv_path: str, model_out: str = "model.joblib"):
    df = pd.read_csv(csv_path)

    X = df[["words", "links", "capital_words", "spam_word_count"]]
    y = df["is_spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, model_out)

    print("Saved model to:", model_out)
    print("\nIntercept:", model.intercept_[0])
    for name, coef in zip(X.columns, model.coef_[0]):
        print(f"Coef({name}) = {coef}")

    print("\nConfusion Matrix:\n", cm)
    print("Accuracy:", acc)


def predict_email(model_path: str, email_text: str):
    model = joblib.load(model_path)
    feats = extract_features(email_text)
    X = pd.DataFrame([feats])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # P(spam)

    print("Extracted features:", feats)
    print("Prediction:", "SPAM" if pred == 1 else "LEGIT")
    print("Spam probability:", float(proba))


def main():
    parser = argparse.ArgumentParser(description="Spam vs Legit classifier (Logistic Regression)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train model from CSV")
    t.add_argument("--csv", required=True, help="Path to CSV dataset")
    t.add_argument("--out", default="model.joblib", help="Output model path")

    p = sub.add_parser("predict", help="Predict from email text")
    p.add_argument("--model", default="model.joblib", help="Path to trained model")
    p.add_argument("--text", required=True, help="Email text to classify")

    args = parser.parse_args()

    if args.cmd == "train":
        train_model(args.csv, args.out)
    elif args.cmd == "predict":
        predict_email(args.model, args.text)


if __name__ == "__main__":
    main()
