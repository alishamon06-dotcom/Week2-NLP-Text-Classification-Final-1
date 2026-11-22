# Week 2 NLP Text Classification Final

This project classifies SMS messages as **Ham (normal message)** or **Spam** using machine-learning models.

## Features
- Downloads the SMS Spam Collection dataset from the UCI repository.
- Cleans the dataset and removes missing values and duplicate messages.
- Converts text to numerical features using TF-IDF.
- Trains and compares Multinomial Naive Bayes and Linear SVM.
- Evaluates models using accuracy, precision, recall, and F1-score.
- Saves confusion-matrix charts, a model comparison CSV, a run summary, and the best trained model.

## Project structure
```text
Week2-NLP-Text-Classification-Final/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── train_classifier.py
├── data/
├── models/
└── reports/
```

## Installation
Run these commands from the project's root folder:

```bash
python -m pip install -r requirements.txt
```

## Run the project
```bash
python src/train_classifier.py
```

The dataset is downloaded automatically the first time the script runs. Generated data, model, and report files are saved in their respective folders.

## Models
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

The best model is selected using the Spam F1-score.

## Dataset
SMS Spam Collection dataset from the UCI Machine Learning Repository.
