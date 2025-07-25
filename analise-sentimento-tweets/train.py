import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve
from transformers import pipeline
from functions import load_data, train_and_evaluate_lr, train_and_evaluate_nb, evaluate_distilbert, split_data, analyze_errors

# Caminhos e constantes
OUTPUT_PATH = "data/embeddings/"
PREPROCESSED_PATH = "data/output/"
INPUT_PATH = "data/input/"
RESULTS = 'data/results/'
EMBEDDING_MODELS = [
    'all-MiniLM-L6-v2',
]
TEXT_COLUMN = "Texto"

X, y, texts = load_data(OUTPUT_PATH, PREPROCESSED_PATH, INPUT_PATH,TEXT_COLUMN)
X_train, X_val, X_test, y_train, y_val, y_test, texts_train, texts_val, texts_test = split_data(X, y, texts)

train_and_evaluate_nb(RESULTS, X_train, y_train, X_val, y_val, X_test, y_test, texts_test)
train_and_evaluate_lr(RESULTS, X_train, y_train, X_val, y_val, X_test, y_test, texts_test, EMBEDDING_MODELS[0])
# evaluate_distilbert(texts_val, y_val, texts_test, y_test)
