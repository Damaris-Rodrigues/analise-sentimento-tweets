import pandas as pd
import os
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
import nltk
from transformers import pipeline
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def process_dataframe(INPUT_PATH, COLUMN_TYPES, TEXT_COLUMN):
    file_path = os.path.join(INPUT_PATH, 'ginastica_2022.xlsx')
    df = pd.read_excel(file_path, dtype=COLUMN_TYPES)

    df.dropna(subset=[TEXT_COLUMN], inplace=True)

    # Remove colunas indesejadas se existirem
    drop_cols = ["Data", "author-id"]
    df.drop(columns=set(drop_cols) & set(df.columns), inplace=True)

    # Stopwords e tokenizer
    stopwords_set = set(stopwords.words('portuguese'))
    tokenizer = TreebankWordTokenizer()

    def preprocess_text(text):
        text = text.lower().strip()
        # Remove links
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove menções (@usuario)
        text = re.sub(r'@\w+', '', text)
        # Remove #, mas mantem o texto
        text = re.sub(r'#', '', text)
        # Remove emojis e símbolos
        text = re.sub(r'[^\w\s]', '', text)  # remove pontuação
        text = re.sub(r'\d+', '', text)      # remove números
        # Tokenização
        tokens = tokenizer.tokenize(text)
        # Remove stopwords e tokens curtos
        tokens = [t for t in tokens if t.isalpha() and t not in stopwords_set and len(t) > 1]
        
        return " ".join(tokens)

    df['texto_processado'] = df[TEXT_COLUMN].apply(preprocess_text)
    return df

def generate_tfidf(df):
    print("Calculando vetores TF-IDF...")
    # Pega a coluna de texto pré-processado, substitui NAs por string vazia e transforma em lista
    texts = df['texto_processado'].fillna("").tolist()

    # Cria o vetorizador TF-IDF com as seguintes configurações:
    vectorizer = TfidfVectorizer(
        max_features=10, # Limita o número de termos mais frequentes
        ngram_range=(1, 2), # Considera unigramas e bigramas (ex: "bom", "dia", "bom dia")
        min_df=5, # Ignora termos que aparecem em menos de 5 documentos
        max_df=0.9 # Ignora termos que aparecem em mais de 90% dos documentos
    )

    # Ajusta o vetor TF-IDF ao texto e transforma em matriz esparsa
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Pega os nomes das características selecionadas
    feature_names = vectorizer.get_feature_names_out()
    # Converte a matriz TF-IDF em DataFrame do pandas, com prefixo nas colunas
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{name}" for name in feature_names])
    # Garante que os índices do DataFrame TF-IDF fiquem alinhados com os do DataFrame original
    df_tfidf.index = df.index
    # Retorna o DataFrame com vetores TF-IDF
    return df_tfidf, vectorizer

def generate_embeddings(df, model_name):
    print(f"Calculando embeddings com o modelo: {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device.upper()}")
    
    # Extrai a coluna 'texto_processado'
    texts = df['texto_processado'].fillna("").to_numpy()
    # Carrega o modelo pré-treinado de embeddings
    model = SentenceTransformer(model_name)
    # Gera os embeddings para todos os textos, com barra de progresso
    embeddings = model.encode(texts, show_progress_bar=True, device=device)
    # Converte os embeddings para um DataFrame, mantendo o mesmo índice do df original
    df_emb = pd.DataFrame(embeddings, index=df.index)
    # Renomeia as colunas do DataFrame para indicar o modelo e a posição do vetor (ex: emb_x0, x1, ..., x767)
    df_emb.columns = [f"{model_name.replace('/', '_')}_x{i}" for i in range(df_emb.shape[1])]

    return df_emb

def classificator(INPUT_PATH, COLUMN_TYPES, BASE_PATH, TEXT_COLUMN):
    file_path = os.path.join(INPUT_PATH, 'ginastica_2022.xlsx')
    df = pd.read_excel(file_path, dtype=COLUMN_TYPES)

    # Verifica se as colunas necessárias existem
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Coluna '{TEXT_COLUMN}' não encontrada no arquivo.")
    if 'ID' not in df.columns:
        raise ValueError("Coluna 'ID' não encontrada no arquivo.")

    # Carrega o modelo de análise de sentimento
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Aplica o modelo nos textos (limitando a 512 caracteres por input)
    texts = df[TEXT_COLUMN].fillna("").tolist()
    results = []
    for text in tqdm(texts, desc="Classificando sentimentos"):
        result = sentiment_analyzer(text[:512])[0]
        results.append(result)

    # Adiciona os resultados ao DataFrame
    df['sentiment_label'] = [r['label'] for r in results]

    # Cria pasta de saída
    OUTPUT_PATH = os.path.join(BASE_PATH, 'output/')
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Salva apenas ID e rótulo
    output_file = os.path.join(OUTPUT_PATH, 'tweets_com_sentimento.xlsx')
    df[['ID', 'sentiment_label']].to_excel(output_file, index=False)

    print(f"\n Arquivo salvo com sentimentos: {output_file}")
        
# ========== Para o treinamento =============

def load_data(OUTPUT_PATH, PREPROCESSED_PATH, INPUT_PATH, TEXT_COLUMN):
    dfx = pd.read_excel(f"{OUTPUT_PATH}TFIDF_e_Embeddings.xlsx")
    dfy = pd.read_excel(f"{PREPROCESSED_PATH}tweets_com_sentimento.xlsx")
    df_original = pd.read_excel(f"{INPUT_PATH}ginastica_2022.xlsx")
    
    X_full = dfx.drop(columns=['ID'])
    y_full = dfy.set_index(dfx.index)['sentiment_label']
    texts_full = df_original.set_index(dfx.index)[TEXT_COLUMN].fillna("").astype(str).tolist()
    
    return X_full, y_full, texts_full


def split_data(X, y, texts, random_state=42):
    # 60% treino, 20% validação, 20% teste
    X_train, X_temp, y_train, y_temp, texts_train, texts_temp = train_test_split(
        X, y, texts, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test, texts_val, texts_test = train_test_split(
        X_temp, y_temp, texts_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, texts_train, texts_val, texts_test


def train_and_evaluate_nb(RESULTS, X_train, y_train, X_val, y_val, X_test, y_test, texts_test):
    print("\nModelo 1: Naive Bayes (TF-IDF)")
    tfidf_cols = [col for col in X_train.columns if col.startswith("tfidf_")]
    
    model = MultinomialNB()
    model.fit(X_train[tfidf_cols], y_train)
    
    # Validação
    y_val_pred = model.predict(X_val[tfidf_cols])
    print("Validação:")
    print(classification_report(y_val, y_val_pred))
    print(f"Acurácia validação: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1-score validação (macro): {f1_score(y_val, y_val_pred, average='macro'):.4f}")
    
    # Teste
    y_test_pred = model.predict(X_test[tfidf_cols])
    print("Teste:")
    print(classification_report(y_test, y_test_pred))
    print(f"Acurácia teste: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"F1-score teste (macro): {f1_score(y_test, y_test_pred, average='macro'):.4f}")

    analyze_errors_nb(RESULTS, model, X_test, y_test, texts_test, tfidf_cols)

def train_and_evaluate_lr(RESULTS, X_train, y_train, X_val, y_val, X_test, y_test, texts_test, embedding_model):
    print(f"\nModelo 2: Logistic Regression ({embedding_model})")
    safe_model_name = embedding_model.replace("/", "-")
    emb_cols = [col for col in X_train.columns if col.startswith(safe_model_name)]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[emb_cols], y_train)
    
    # Validação
    y_val_pred = model.predict(X_val[emb_cols])
    print("Validação:")
    print(classification_report(y_val, y_val_pred))
    print(f"Acurácia validação: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1-score validação (macro): {f1_score(y_val, y_val_pred, average='macro'):.4f}")
    
    # Teste
    y_test_pred = model.predict(X_test[emb_cols])
    y_test_proba = model.predict_proba(X_test[emb_cols])
    print("Teste:")
    print(classification_report(y_test, y_test_pred))
    print(f"Acurácia teste: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"F1-score teste (macro): {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    
    # ROC-AUC
    if 'POSITIVE' in model.classes_:
        pos_idx = list(model.classes_).index('POSITIVE')
        y_test_bin = y_test.map({'NEGATIVE': 0, 'POSITIVE': 1})
        roc_auc = roc_auc_score(y_test_bin, y_test_proba[:, pos_idx])
        print(f"ROC-AUC teste: {roc_auc:.4f}")
        fpr, tpr, _ = roc_curve(y_test_bin, y_test_proba[:, pos_idx])
        plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("Curva ROC - Logistic Regression")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    analyze_errors(RESULTS, model, X_test, y_test, texts_test, emb_cols, safe_model_name)

def evaluate_distilbert(texts_val, y_val, texts_test, y_test):
    print("\nModelo 3: DistilBERT (pré-treinado Hugging Face)")
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Validação
    preds_val = [sentiment_model(text[:512])[0]['label'] for text in texts_val]
    print("Validação:")
    print(classification_report(y_val.tolist(), preds_val))
    print(f"Acurácia validação: {accuracy_score(y_val.tolist(), preds_val):.4f}")
    print(f"F1-score validação (macro): {f1_score(y_val.tolist(), preds_val, average='macro'):.4f}")
    
    # Teste
    preds_test = [sentiment_model(text[:512])[0]['label'] for text in texts_test]
    print("Teste:")
    print(classification_report(y_test.tolist(), preds_test))
    print(f"Acurácia teste: {accuracy_score(y_test.tolist(), preds_test):.4f}")
    print(f"F1-score teste (macro): {f1_score(y_test.tolist(), preds_test, average='macro'):.4f}")

#----------- Analise de erros ----------
def analyze_errors(RESULTS, model, X_test, y_test, texts_test, emb_cols, model_name):
    y_pred = model.predict(X_test[emb_cols])
    errors = []

    for i in range(len(y_test)):
        if y_pred[i] != y_test.iloc[i]:
            errors.append({
                "Texto": texts_test[i],
                "Verdadeiro": y_test.iloc[i],
                "Predito": y_pred[i]
            })

    # Exibe os 10 primeiros erros
    print(f"\nExemplos de erros de classificação ({model_name}):")
    for e in errors[:10]:
    # Salva em excel
        errors_df = pd.DataFrame(errors)
    errors_df.to_excel(os.path.join(RESULTS, f"errors_{model_name}.xlsx"), index=False)
    print(f"\nArquivo com erros salvo em: errors_{model_name}.xlsx")
    
def analyze_errors_nb(RESULTS, model, X_test, y_test, texts_test, tfidf_cols):
    y_pred = model.predict(X_test[tfidf_cols])
    errors = []

    for i in range(len(y_test)):
        if y_pred[i] != y_test.iloc[i]:
            errors.append({
                "Texto": texts_test[i],
                "Verdadeiro": y_test.iloc[i],
                "Predito": y_pred[i]
            })

    errors_df = pd.DataFrame(errors)
    errors_df.to_excel(os.path.join(RESULTS, "errors_nb.xlsx"), index=False)
    print(f"\nArquivo com erros (Naive Bayes) salvo: errors_nb.xlsx")
