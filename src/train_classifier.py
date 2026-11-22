from pathlib import Path
import urllib.request
import zipfile
import io
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

for folder in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset():
    if DATA_FILE.exists():
        print("Dataset already exists.")
        return

    print("Downloading dataset from UCI...")
    request = urllib.request.Request(
        DATASET_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
            matching_files = [
                name for name in archive.namelist()
                if Path(name).name == "SMSSpamCollection"
            ]
            if not matching_files:
                raise FileNotFoundError("SMSSpamCollection not found in downloaded ZIP.")
            DATA_FILE.write_bytes(archive.read(matching_files[0]))

        print("Dataset downloaded successfully.")
    except Exception as error:
        raise RuntimeError(f"Dataset download failed: {error}") from error


def load_dataset():
    print("\nLoading dataset...")
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8"
    )

    print("Original records:", len(df))
    df.dropna(subset=["label", "text"], inplace=True)
    df.drop_duplicates(inplace=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["text"] != ""]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    print("Records after cleaning:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())
    df.to_csv(DATA_DIR / "cleaned_sms_dataset.csv", index=False)
    return df


def split_data(df):
    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))
    return X_train, X_test, y_train, y_test


def create_models():
    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", MultinomialNB())
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42))
        ])
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    best_model, best_score, best_name = None, -1, None

    for name, model in models.items():
        print("\n" + "=" * 45)
        print("Training:", name)
        print("=" * 45)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label="spam", zero_division=0)
        recall = recall_score(y_test, predictions, pos_label="spam", zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label="spam", zero_division=0)

        print("Accuracy:", round(accuracy, 4))
        print("Spam Precision:", round(precision, 4))
        print("Spam Recall:", round(recall, 4))
        print("Spam F1 Score:", round(f1, 4))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Spam Precision": precision,
            "Spam Recall": recall,
            "Spam F1 Score": f1
        })

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, labels=["ham", "spam"],
            display_labels=["Ham", "Spam"], cmap="Blues"
        )
        disp.ax_.set_title(name)
        chart_name = name.lower().replace(" ", "_").replace("multinomial_", "")
        plt.savefig(REPORT_DIR / f"{chart_name}_confusion_matrix.png", bbox_inches="tight")
        plt.close()

        if f1 > best_score:
            best_score, best_model, best_name = f1, model, name

    return results, best_model, best_name


def save_results(results, best_model, best_name):
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_model, MODEL_DIR / "best_text_classifier.joblib")

    with (REPORT_DIR / "run_summary.txt").open("w", encoding="utf-8") as file:
        file.write("WEEK 2 NLP TEXT CLASSIFICATION RESULTS\n")
        file.write("=" * 40 + "\n\n")
        file.write(results_df.to_string(index=False))
        file.write(f"\n\nBest Model: {best_name}\n")
        file.write("Selection Metric: Spam F1 Score\n")

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))
    print("\nBest Model:", best_name)
    print("All results saved successfully.")


def main():
    print("\nWEEK 2 NLP TEXT CLASSIFICATION")
    print("=" * 40)
    download_dataset()
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    models = create_models()
    results, best_model, best_name = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )
    save_results(results, best_model, best_name)
    print("\nPROJECT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nPROJECT FAILED!")
        print("Error:", error)
        raise
