import pandas as pd
import os
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

RANDOM_STATE = 42

def process_dataframe(INPUT_PATH, COLUMN_TYPES, TEXT_COLUMN):
    file_path = os.path.join(INPUT_PATH, 'ginastica_2022.xlsx')
    df = pd.read_excel(file_path, dtype=COLUMN_TYPES)

    df.dropna(subset=[TEXT_COLUMN], inplace=True)

    # Remove colunas indesejadas se existirem
    drop_cols = ["Data", "author-id"]
    df.drop(columns=set(drop_cols) & set(df.columns), inplace=True)

    # Stopwords e tokenizer
    stopwords_set = set(stopwords.words('english'))
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
        max_features=500, # Limita o número de termos mais frequentes
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
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

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


def train_and_evaluate_nb_binary(RESULTS, X_train, y_train, X_val, y_val, X_test, y_test, texts_test):
    print("\nModelo 1: Naive Bayes (TF-IDF) - Binário (POSITIVE vs NEGATIVE)")
    y_train = y_train.str.upper()
    y_val = y_val.str.upper()
    y_test = y_test.str.upper()
    
    # Filtra apenas POSITIVE e NEGATIVE
    mask_train = y_train.isin(["POSITIVE", "NEGATIVE"])
    mask_val = y_val.isin(["POSITIVE", "NEGATIVE"])
    mask_test = y_test.isin(["POSITIVE", "NEGATIVE"])

    X_train_bin = X_train[mask_train]
    y_train_bin = y_train[mask_train]
    X_val_bin = X_val[mask_val]
    y_val_bin = y_val[mask_val]
    X_test_bin = X_test[mask_test]
    y_test_bin = y_test[mask_test]
    texts_test_bin = [t for t, keep in zip(texts_test, mask_test) if keep]

    tfidf_cols = [col for col in X_train_bin.columns if col.startswith("tfidf_")]

    best_alpha = None
    best_f1 = -1
    best_model = None
    alphas = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    print(" Ajustando hiperparâmetro alpha...")
    for alpha in alphas:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train_bin[tfidf_cols], y_train_bin)
        y_val_pred = model.predict(X_val_bin[tfidf_cols])
        f1 = f1_score(y_val_bin, y_val_pred, average='macro')
        print(f"alpha={alpha:.2f} -> F1 (macro): {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_model = model

    print(f"\n Melhor alpha encontrado: {best_alpha} com F1 (macro) = {best_f1:.4f}")

    # Validação
    print("\nValidação:")
    y_val_pred = best_model.predict(X_val_bin[tfidf_cols])
    print(classification_report(y_val_bin, y_val_pred))
    print(f"Acurácia validação: {accuracy_score(y_val_bin, y_val_pred):.4f}")
    print(f"F1-score validação (macro): {f1_score(y_val_bin, y_val_pred, average='macro'):.4f}")

    # Teste
    print("\nTeste:")
    y_test_pred = best_model.predict(X_test_bin[tfidf_cols])
    print(classification_report(y_test_bin, y_test_pred))
    print(f"Acurácia teste: {accuracy_score(y_test_bin, y_test_pred):.4f}")
    print(f"F1-score teste (macro): {f1_score(y_test_bin, y_test_pred, average='macro'):.4f}")

    if set(best_model.classes_) == {"POSITIVE", "NEGATIVE"}:
        pos_idx = list(best_model.classes_).index('POSITIVE')
        neg_idx = list(best_model.classes_).index('NEGATIVE')

        y_test_bin_pos = y_test_bin.map({'NEGATIVE': 0, 'POSITIVE': 1})
        y_test_bin_neg = y_test_bin.map({'NEGATIVE': 1, 'POSITIVE': 0})

        y_test_proba = best_model.predict_proba(X_test_bin[tfidf_cols])

        # ROC-AUC Positive
        roc_auc_pos = roc_auc_score(y_test_bin_pos, y_test_proba[:, pos_idx])
        fpr_pos, tpr_pos, _ = roc_curve(y_test_bin_pos, y_test_proba[:, pos_idx])

        # ROC-AUC Negative
        roc_auc_neg = roc_auc_score(y_test_bin_neg, y_test_proba[:, neg_idx])
        fpr_neg, tpr_neg, _ = roc_curve(y_test_bin_neg, y_test_proba[:, neg_idx])

        plt.figure(figsize=(8,6))
        plt.plot(fpr_pos, tpr_pos, label=f"POSITIVE (AUC = {roc_auc_pos:.2f})")
        plt.plot(fpr_neg, tpr_neg, label=f"NEGATIVE (AUC = {roc_auc_neg:.2f})", linestyle='--')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("Curvas ROC - Naive Bayes (POSITIVE vs NEGATIVE)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    if 'analyze_errors_nb' in globals():
        analyze_errors_nb(RESULTS, best_model, X_test_bin, y_test_bin, texts_test_bin, tfidf_cols)

    return best_model

def train_and_evaluate_lr(RESULTS, X_train, y_train, X_val, y_val, X_test, y_test, texts_test, embedding_model):
    print(f"\nModelo 2: Logistic Regression ({embedding_model})")
    safe_model_name = embedding_model.replace("/", "-")
    emb_cols = [col for col in X_train.columns if col.startswith(safe_model_name)]

    Cs = [0.01, 0.1, 1.0, 2.0]
    penalties = ['l1', 'l2']
    best_f1 = -1
    best_params = None
    best_model = None

    for penalty in penalties:
        for C in Cs:
            try:
                model = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    max_iter=1000,
                    solver='saga'
                )
                model.fit(X_train[emb_cols], y_train)
                y_val_pred = model.predict(X_val[emb_cols])
                f1 = f1_score(y_val, y_val_pred, average='macro')

                print(f"Teste: penalty={penalty}, C={C} -> F1 val (macro): {f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'penalty': penalty, 'C': C}
                    best_model = model
            except Exception as e:
                print(f"Erro com penalty={penalty}, C={C}: {e}")

    print(f"\nMelhor modelo: penalty={best_params['penalty']}, C={best_params['C']} com F1 val (macro)={best_f1:.4f}")

    # Avaliação no conjunto de validação (melhor modelo)
    print("\nValidação (melhor modelo):")
    y_val_pred = best_model.predict(X_val[emb_cols])
    print(classification_report(y_val, y_val_pred))
    print(f"Acurácia validação: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1-score validação (macro): {f1_score(y_val, y_val_pred, average='macro'):.4f}")

    # Avaliação no conjunto de teste
    print("\nTeste (melhor modelo):")
    y_test_pred = best_model.predict(X_test[emb_cols])
    y_test_proba = best_model.predict_proba(X_test[emb_cols])
    print(classification_report(y_test, y_test_pred))
    print(f"Acurácia teste: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"F1-score teste (macro): {f1_score(y_test, y_test_pred, average='macro'):.4f}")

    # Plot curva ROC para 2 ou mais classes
    classes = best_model.classes_
    y_test_proba = best_model.predict_proba(X_test[emb_cols])

    if len(classes) == 2:
        # Caso binário (mantém lógica original)
        pos_idx = list(classes).index('POSITIVE')
        neg_idx = list(classes).index('NEGATIVE')

        y_test_bin_pos = y_test.map({'NEGATIVE': 0, 'POSITIVE': 1})
        y_test_bin_neg = y_test.map({'NEGATIVE': 1, 'POSITIVE': 0})

        roc_auc_pos = roc_auc_score(y_test_bin_pos, y_test_proba[:, pos_idx])
        fpr_pos, tpr_pos, _ = roc_curve(y_test_bin_pos, y_test_proba[:, pos_idx])

        roc_auc_neg = roc_auc_score(y_test_bin_neg, y_test_proba[:, neg_idx])
        fpr_neg, tpr_neg, _ = roc_curve(y_test_bin_neg, y_test_proba[:, neg_idx])

        plt.figure(figsize=(8,6))
        plt.plot(fpr_pos, tpr_pos, label=f"POSITIVE (AUC = {roc_auc_pos:.2f})")
        plt.plot(fpr_neg, tpr_neg, label=f"NEGATIVE (AUC = {roc_auc_neg:.2f})", linestyle='--')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("Curvas ROC - Logistic Regression (Binário)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # Caso multi-classe (One-vs-Rest)
        y_bin = label_binarize(y_test, classes=classes)

        plt.figure(figsize=(8,6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_test_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("Curvas ROC - Logistic Regression (Multi-classe OvR)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Análise de erros
    analyze_errors(RESULTS, best_model, X_test, y_test, texts_test, emb_cols, safe_model_name)

    return best_model

def evaluate_distilbert_binary(texts_val, y_val, texts_test, y_test):
    print("\nModelo 3: DistilBERT (binário POSITIVE vs NEGATIVE)")

    y_val = y_val.str.upper()
    y_test = y_test.str.upper()
    
    # Filtra apenas POSITIVE e NEGATIVE
    mask_val = y_val.isin(["POSITIVE", "NEGATIVE"])
    mask_test = y_test.isin(["POSITIVE", "NEGATIVE"])

    texts_val_bin = [t for t, keep in zip(texts_val, mask_val) if keep]
    y_val_bin = y_val[mask_val]

    texts_test_bin = [t for t, keep in zip(texts_test, mask_test) if keep]
    y_test_bin = y_test[mask_test]

    sentiment_model = pipeline("sentiment-analysis", 
                               model="distilbert-base-uncased-finetuned-sst-2-english",
                               truncation=True,
                               device=0 if torch.cuda.is_available() else -1,
                               batch_size=32)

    def normalize_label(label):
        return label.upper()

    # ----- Validação -----
    preds_val_raw = sentiment_model(texts_val_bin)
    preds_val = [normalize_label(p['label']) for p in preds_val_raw]
    y_val_norm = [normalize_label(label) for label in y_val_bin]

    print("Validação:")
    print(classification_report(y_val_norm, preds_val))
    print(f"Acurácia validação: {accuracy_score(y_val_norm, preds_val):.4f}")
    print(f"F1-score validação (macro): {f1_score(y_val_norm, preds_val, average='macro'):.4f}")

    # ----- Teste -----
    preds_test_raw = sentiment_model(texts_test_bin)
    preds_test = [normalize_label(p['label']) for p in preds_test_raw]
    y_test_norm = [normalize_label(label) for label in y_test_bin]

    print("Teste:")
    print(classification_report(y_test_norm, preds_test))
    print(f"Acurácia teste: {accuracy_score(y_test_norm, preds_test):.4f}")
    print(f"F1-score teste (macro): {f1_score(y_test_norm, preds_test, average='macro'):.4f}")

    # ----- Curvas ROC POSITIVE e NEGATIVE -----
    # Probabilidades da classe POSITIVE
    y_test_pos_prob = [p['score'] if p['label'] == 'POSITIVE' else 1 - p['score'] for p in preds_test_raw]
    y_test_bin_pos = [1 if y == "POSITIVE" else 0 for y in y_test_norm]

    # Probabilidades da classe NEGATIVE
    y_test_neg_prob = [p['score'] if p['label'] == 'NEGATIVE' else 1 - p['score'] for p in preds_test_raw]
    y_test_bin_neg = [1 if y == "NEGATIVE" else 0 for y in y_test_norm]

    # ROC POSITIVE
    roc_auc_pos = roc_auc_score(y_test_bin_pos, y_test_pos_prob)
    fpr_pos, tpr_pos, _ = roc_curve(y_test_bin_pos, y_test_pos_prob)

    # ROC NEGATIVE
    roc_auc_neg = roc_auc_score(y_test_bin_neg, y_test_neg_prob)
    fpr_neg, tpr_neg, _ = roc_curve(y_test_bin_neg, y_test_neg_prob)

    # Plotando as curvas
    plt.figure(figsize=(8,6))
    plt.plot(fpr_pos, tpr_pos, label=f"POSITIVE (AUC = {roc_auc_pos:.2f})")
    plt.plot(fpr_neg, tpr_neg, label=f"NEGATIVE (AUC = {roc_auc_neg:.2f})", linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curvas ROC - DistilBERT")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

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
