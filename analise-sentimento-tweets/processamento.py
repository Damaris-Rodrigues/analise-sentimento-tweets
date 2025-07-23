import pandas as pd
import os
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve
from functions import process_dataframe, generate_embeddings, generate_tfidf, classificator
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
import matplotlib.pyplot as plt

# Configurações
BASE_PATH = 'data/'
INPUT_PATH = os.path.join(BASE_PATH, 'input/')
OUTPUT_PATH = os.path.join(BASE_PATH, 'embeddings/')
PREPROCESSED_PATH = BASE_PATH + 'output/'
EMBEDDING_MODELS = [
    'google-bert/bert-base-uncased',
]
TEXT_COLUMN = 'Texto'
COLUMN_TYPES = {'ID': str}
df_original = pd.read_excel(os.path.join(INPUT_PATH, "ginastica_2022.xlsx"), dtype={"ID": str})

if __name__ == "__main__":
    print("\n Processando tweets:")
    
    #Pre processa os dados
    df_process = process_dataframe(INPUT_PATH, COLUMN_TYPES, TEXT_COLUMN)
    
    '''# Classifica em sentimentos negativos ou positivos (NÃO MEXER AQUI, POIS ESTÁ CLASSIFICADO)
    df_class = classificator(INPUT_PATH, COLUMN_TYPES, TEXT_COLUMN, TEXT_COLUMN)'''
    
    # Gera TF-IDF
    print("\n Gerando TF-IDF:")
    df_tfidf, tfidf_vectorizer = generate_tfidf(df_process)

    # Embeddings
    print("\n Gerando embeddings:")
    all_embeddings = []
    for model_name in tqdm(EMBEDDING_MODELS, desc='Modelos'):
        cache_path = os.path.join(OUTPUT_PATH, f"embedding_{model_name.replace('/', '_')}.csv")

        if os.path.exists(cache_path):
            print(f" Embedding já existente: {cache_path}")
            df_emb = pd.read_csv(cache_path)
            df_emb.index = df_process.index
        else:
            df_emb = generate_embeddings(df_process, model_name)
            df_emb.to_csv(cache_path, index=False)

        all_embeddings.append(df_emb)

    # Combina tudo
    df_final = pd.concat(
        [df_process[['ID']]] + [df_tfidf] + all_embeddings,
        axis=1
    )

    # Salva com timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    final_file = os.path.join(OUTPUT_PATH, f'TFIDF_e_Embeddings.xlsx')
    df_final.to_excel(final_file, index=False)
    print(f"\n Arquivo final salvo: {final_file}")