"""Preparação dos dados compartilhada pelos notebooks de regressão e classificação.

Centraliza a leitura do dataset limpo, o split treino/teste e o pré-processamento,
para que as duas partes do trabalho usem exatamente o mesmo tratamento.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Semente fixa: o split e os modelos saem iguais em toda execução
RANDOM_STATE = 42

# Caminho calculado a partir deste arquivo, então funciona de qualquer pasta
CAMINHO_DADOS = Path(__file__).resolve().parents[1] / "data" / "processed" / "winequality_clean.csv"

# Variáveis com cauda longa (seção 5 da análise exploratória): candidatas ao log
COLUNAS_CAUDA_LONGA = ["residual sugar", "chlorides", "free sulfur dioxide", "sulphates"]

COLUNAS_NUMERICAS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]


def carregar_dados():
    """Lê o dataset limpo e separa as features (X) do target (y)."""
    df = pd.read_csv(CAMINHO_DADOS)
    X = df.drop(columns="quality")
    y = df["quality"]
    return X, y


def dividir_treino_teste(X, y, test_size=0.2):
    """Split 80/20 estratificado pela quality, para as notas raras irem aos dois conjuntos."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def criar_pipeline(modelo):
    """Monta o pipeline completo: pré-processamento + modelo.

    Etapas:
    1. "prep": trata cada grupo de colunas separadamente
       - "cauda": mediana nos vazios e, opcionalmente, log (desligado por padrão)
       - "num":   mediana nos vazios
       - "type":  red/white vira uma coluna 0/1
    2. "scaler": padroniza todas as colunas
    3. "model":  o modelo recebido

    Os nomes das etapas são usados no GridSearchCV para trocar opções, por exemplo:
    - "prep__cauda__log": "passthrough" (sem log) ou FunctionTransformer(np.log1p)
    - "scaler":           StandardScaler() ou RobustScaler()
    - "model__alpha":     hiperparâmetro do modelo
    """
    outras_numericas = [c for c in COLUNAS_NUMERICAS if c not in COLUNAS_CAUDA_LONGA]

    preprocessador = ColumnTransformer([
        ("cauda", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", "passthrough"),   # "passthrough" = etapa desligada
        ]), COLUNAS_CAUDA_LONGA),
        ("num", SimpleImputer(strategy="median"), outras_numericas),
        ("type", OneHotEncoder(drop="if_binary"), ["type"]),
    ], verbose_feature_names_out=False)   # mantém os nomes originais das colunas

    return Pipeline([
        ("prep", preprocessador),
        ("scaler", StandardScaler()),
        ("model", modelo),
    ])
