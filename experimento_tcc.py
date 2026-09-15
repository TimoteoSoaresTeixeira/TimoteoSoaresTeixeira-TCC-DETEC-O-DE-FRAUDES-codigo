import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

caminho_arquivo = 'train_transaction.csv'

print(f"\n{'='*60}\nIniciando processamento da base IEEE-CIS Fraud Detection\n{'='*60}")

if not os.path.exists(caminho_arquivo):
    print(f"⚠️ ERRO: O arquivo '{caminho_arquivo}' não foi encontrado na pasta.")
else:
    # 1. Carregar a base de dados (Limitando a 100 mil linhas para não travar o PC)
    print("Carregando o dataset (Amostra de 100.000 transações)...")
    df = pd.read_csv(caminho_arquivo, nrows=100000)

    # 2. Pré-processamento e Limpeza (Vida Real)
    print("Limpando dados vazios e convertendo textos em números...")
    
    # Separar a variável alvo
    y = df['isFraud']
    
    # Remover colunas inúteis para o aprendizado e o próprio alvo
    X = df.drop(['TransactionID', 'isFraud', 'TransactionDT'], axis=1)

    # ========================================================
    # CORREÇÃO: Limpeza blindada contra tipos de dados mistos
    # ========================================================
    for col in X.columns:
        # Se for realmente uma coluna puramente numérica
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median()) # Preenche com a mediana
        else:
            # Se for texto (string, object, category) ou misturado
            X[col] = X[col].fillna('Desconhecido') # Preenche vazios
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str)) # Força virar string e depois converte para número
    # ========================================================

    # Padronização
    print("Padronizando os dados...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Particionamento Holdout (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Cálculo do peso para o XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Definir modelos
    models = {
        'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=42)
    }

    models_class_weight = {
        'Regressão Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42)
    }

    samplers = {
        'Baseline (Sem Trat)': None,
        'Under-Sampling': RandomUnderSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42)
    }

    resultados = []
    print("Iniciando o treinamento dos modelos. Isso pode levar alguns minutos...\n")

    # 4. Loop de Treinamento
    for sampler_name, sampler in samplers.items():
        if sampler is not None:
            print(f"-> Aplicando {sampler_name}...")
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train

        for model_name, model in models.items():
            print(f"   Treinando: {model_name}...")
            
            start_train = time.time()
            model.fit(X_resampled, y_resampled)
            end_train = time.time()
            
            start_pred = time.time()
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            end_pred = time.time()
            
            resultados.append({
                'Modelo': model_name,
                'Técnica': sampler_name,
                'Precisão': round(precision_score(y_test, y_pred), 4),
                'Revocação': round(recall_score(y_test, y_pred), 4),
                'F1-Score': round(f1_score(y_test, y_pred), 4),
                'AUC-ROC': round(roc_auc_score(y_test, y_prob), 4),
                'AUC-PR': round(average_precision_score(y_test, y_prob), 4),
                'Tempo Treino (s)': round(end_train - start_train, 3)
            })

    # Modelos com Class Weight
    print("-> Aplicando Class Weight...")
    for model_name, model in models_class_weight.items():
        print(f"   Treinando: {model_name}...")
        
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        resultados.append({
            'Modelo': model_name,
            'Técnica': 'Class Weight',
            'Precisão': round(precision_score(y_test, y_pred), 4),
            'Revocação': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'AUC-ROC': round(roc_auc_score(y_test, y_prob), 4),
            'AUC-PR': round(average_precision_score(y_test, y_prob), 4),
            'Tempo Treino (s)': round(end_train - start_train, 3)
        })

    # 5. Salvar e Gerar Gráficos
    df_resultados = pd.DataFrame(resultados)
    
    print("\n" + "="*60)
    print(" RESULTADOS FINAIS - IEEE-CIS FRAUD DETECTION ")
    print("="*60)
    print(df_resultados.to_string(index=False))

    df_resultados.to_excel('resultados_ieee_cis.xlsx', index=False)
    
    # Gráficos
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Eficiência Preditiva (AUC-PR)
    plt.figure(figsize=(10, 6))
    g1 = sns.barplot(data=df_resultados, x="Técnica", y="AUC-PR", hue="Modelo", palette="viridis")
    plt.title('Eficiência Preditiva (AUC-PR) - IEEE-CIS', fontsize=14)
    plt.tight_layout()
    plt.savefig('ieee_auc_pr.png', dpi=300)

    # Gráfico 2: F1-Score
    plt.figure(figsize=(10, 6))
    g2 = sns.barplot(data=df_resultados, x="Técnica", y="F1-Score", hue="Modelo", palette="magma")
    plt.title('Equilíbrio de Acertos (F1-Score) - IEEE-CIS', fontsize=14)
    plt.tight_layout()
    plt.savefig('ieee_f1_score.png', dpi=300)

    print("\n✅ Sucesso! Planilha 'resultados_ieee_cis.xlsx' e gráficos gerados na sua pasta.")