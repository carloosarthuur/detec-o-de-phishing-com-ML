import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    # Carrega o arquivo CSV gerado na Fase 3
    df = pd.read_csv('features.csv')
    print("Arquivo 'features.csv' carregado com sucesso.")
    print(f"O dataset contém {df.shape[0]} amostras e {df.shape[1]} colunas.")
except FileNotFoundError:
    print("Erro: Arquivo 'features.csv' não encontrado. Certifique-se de que ele está na mesma pasta que este script.")
    exit()

# --- Tratamento de Dados Faltantes ---
# O script de features pode ter gerado valores nulos ou -1 para atributos
# que não puderam ser extraídos. Vamos preenchê-los com a mediana da coluna.
for col in df.columns:
    if df[col].dtype == 'float64' or df[col].dtype == 'int64':
        median = df[col].median()
        df[col] = df[col].fillna(median)
        df[col] = df[col].replace(-1, median)


# --- 2. Redução de Atributos (conforme o artigo) ---
# O estudo removeu 3 atributos específicos.
# [cite_start]A16 e A20 foram removidos por possuírem variância nula no dataset original. [cite: 159]
# [cite_start]A11 foi removido por ter uma correlação linear negativa perfeita com A2. [cite: 161]

# Criamos uma lista dos atributos a serem removidos
atributos_para_remover = ['A11', 'A16', 'A20']
df_limpo = df.drop(columns=atributos_para_remover, errors='ignore') # 'errors=ignore' evita erro caso a coluna já tenha sido removida

print(f"\nAtributos removidos: {atributos_para_remover}")
print(f"O dataset agora contém {df_limpo.shape[1]} colunas.")


# --- 3. Separar Features (X) e Rótulos (y) ---
# O rótulo (label) é o que queremos prever (0 ou 1).
# As features são todos os outros atributos, exceto a URL que é apenas um identificador.
X = df_limpo.drop(columns=['label', 'url'])
y = df_limpo['label']

print("\nDados separados em features (X) e rótulos (y).")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")


# --- 4. Normalização dos Dados ---
# [cite_start]O estudo normalizou os dados para que todos os valores ficassem no intervalo de 0 a 1. [cite: 207]
# Isso é importante para que algoritmos sensíveis à escala, como o KNN e SVM, funcionem corretamente.

# Instancia o normalizador
scaler = MinMaxScaler()

# Aplica a normalização nas features
X_normalized = scaler.fit_transform(X)

print("\nFeatures normalizadas com sucesso.")

# Para visualização, podemos criar um DataFrame com os dados normalizados
df_normalized = pd.DataFrame(X_normalized, columns=X.columns)
print("Amostra dos dados normalizados (primeiras 5 linhas):")
print(df_normalized.head())


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Importação dos modelos
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print("\n--- Iniciando Treinamento e Avaliação ---")

# --- 1. Definição dos Modelos com Parâmetros do Artigo ---
# Os parâmetros foram extraídos da Tabela 2 do documento.

models = {
    "MLP": MLPClassifier(
        solver='sgd',
        activation='tanh',
        learning_rate_init=0.075,
        hidden_layer_sizes=(25, 40, 10),
        validation_fraction=0.2,
        max_iter=500,
        random_state=42 # Adicionado para reprodutibilidade
    ),
    "SVM": SVC(
        kernel='rbf',
        gamma='scale',
        tol=0.001,
        decision_function_shape='ovr',
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=3,
        weights='uniform',
        metric='euclidean',
        algorithm='auto'
    ),
    "Árvore de Decisão": DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    ),
    # O Naive Bayes é um caso especial que usa PCA.
    "Naive Bayes": Pipeline([
        ('pca', PCA(n_components=15)), #15 para versão completa e 5 para versão simplificada
        ('naive_bayes', GaussianNB())
    ])
}

# --- 2. Configuração da Validação Cruzada (k=5) ---
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Métricas que serão calculadas, conforme o artigo.
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

# Dicionário para armazenar os resultados
results = {}

# --- 3. Execução do Treinamento e Avaliação ---
print("Avaliando os modelos com validação cruzada...")
for model_name, model in models.items():
    # A função cross_validate treina e avalia o modelo 5 vezes
    scores = cross_validate(model, X_normalized, y, cv=cv_strategy, scoring=scoring_metrics)
    results[model_name] = scores
    print(f"- {model_name}: Avaliação concluída.")

# --- 4. Exibição dos Resultados ---
print("\n--- Resultados Finais (Média das 5 execuções) ---")

#DataFrame para exibir os resultados de forma organizada
summary = []
for model_name, scores in results.items():
    summary.append({
        'Algoritmo': model_name,
        'Acurácia': np.mean(scores['test_accuracy']),
        'Precisão': np.mean(scores['test_precision']),
        'Recall': np.mean(scores['test_recall']),
        'F1-Score': np.mean(scores['test_f1'])
    })

df_results = pd.DataFrame(summary)
print(df_results.to_string(index=False))