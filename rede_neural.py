import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

caminho_arquivo = 'train_transaction.csv'

print(f"\n{'='*60}\nIniciando: Rede Neural Profunda (100% ORGÂNICO E ALEATÓRIO)\n{'='*60}")

# 1. Carregando e Limpando os Dados
df = pd.read_csv(caminho_arquivo, nrows=100000)
y = df['isFraud']
X = df.drop(['TransactionID', 'isFraud', 'TransactionDT'], axis=1)

print("Realizando faxina nos dados...")
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna('Desconhecido')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SEM MANIPULAÇÃO: O sorteio de quem vai para treino ou teste agora é real!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

# ========================================================
# 2. A REDE NEURAL PROFUNDA (ESTOCÁSTICA)
# ========================================================
print("\nTreinando a Rede Neural Viva (Pesos neurais aleatórios)...")
# SEM MANIPULAÇÃO: Sem random_state. A IA aprende do zero de verdade agora.
rede_neural = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=50)

start_time = time.time()
rede_neural.fit(X_train, y_train)
tempo_treino = time.time() - start_time

y_prob = rede_neural.predict_proba(X_test)[:, 1]

# Limiar rígido: 20%
y_pred_rigido = (y_prob >= 0.2).astype(int)

# ========================================================
# 3. EXTRAÇÃO DE MÉTRICAS E GERAÇÃO DE ARQUIVOS
# ========================================================
cm = confusion_matrix(y_test, y_pred_rigido)
verdadeiros_negativos = cm[0][0]
falsos_positivos = cm[0][1]      # Inocentes bloqueados
falsos_negativos = cm[1][0]      # Fraudes que passaram
verdadeiros_positivos = cm[1][1] # Fraudes pegas

auc_pr = average_precision_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred_rigido)

print(f"\nRESULTADOS ORGÂNICOS DESTA EXECUÇÃO:")
print(f"-> Fraudes pegas: {verdadeiros_positivos}")
print(f"-> Inocentes bloqueados: {falsos_positivos}")
print(f"(Rode o código novamente e veja que a IA aprende diferente a cada vez!)\n")

# --- ARQUIVO 1: O EXCEL ---
dados_excel = {
    'Modelo': ['Rede Neural (MLP)'],
    'Limiar (Corte)': ['Rígido (20%)'],
    'AUC-PR': [round(auc_pr, 4)],
    'F1-Score': [round(f1, 4)],
    'Fraudes Pegas (VP)': [verdadeiros_positivos],
    'Inocentes Bloqueados (FP)': [falsos_positivos],
    'Tempo Treino (s)': [round(tempo_treino, 2)]
}
df_resultados = pd.DataFrame(dados_excel)
df_resultados.to_excel('resultados_rede_neural.xlsx', index=False)

# --- ARQUIVO 2: O GRÁFICO DA MATRIZ DE CONFUSÃO ---
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Legítimo', 'Fraude'],
            yticklabels=['Legítimo', 'Fraude'], annot_kws={"size": 14})
plt.title('Matriz de Confusão - O Custo Social', fontsize=14, pad=15)
plt.ylabel('Acontecimento Real', fontsize=12)
plt.xlabel('Previsão da IA', fontsize=12)
plt.tight_layout()
plt.savefig('matriz_confusao_ia.png', dpi=300)
plt.close()

# --- ARQUIVO 3: O GRÁFICO DE DISTRIBUIÇÃO ---
plt.figure(figsize=(10, 6))
prob_legitimos = y_prob[y_test == 0]
prob_fraudes = y_prob[y_test == 1]

sns.kdeplot(prob_legitimos, color="blue", fill=True, alpha=0.3, label="Clientes Legítimos")
sns.kdeplot(prob_fraudes, color="red", fill=True, alpha=0.3, label="Fraudes Reais")

plt.axvline(x=0.2, color='black', linestyle='--', linewidth=2, label='Corte Rígido (20%)')

plt.title('Por que a IA erra? O conflito entre o Inocente e o Fraude', fontsize=15, pad=15)
plt.xlabel('Probabilidade calculada pelo Algoritmo (%)', fontsize=12)
plt.ylabel('Densidade / Volume de Transações', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('distribuicao_probabilidades_ia.png', dpi=300)
plt.close()

print(f"Arquivos gerados com sucesso na sua pasta!")
print("="*60)