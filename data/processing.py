import pandas as pd
import numpy as np
import os

# =========================================================
# 1. Leitura dos dados brutos
# =========================================================
# dtype=str: lê "alcohol" como texto para tratar os valores corrompidos
df = pd.read_csv("data/raw/group_3_winequality.csv", dtype={"alcohol": str})


# =========================================================
# 2. Coluna "quality" (o que o modelo vai prever)
# =========================================================
print(df["quality"].value_counts(dropna=False).sort_index())

# Mantém só as notas entre 0 e 10.
# Linhas vazias também são removidas aqui, porque vazio não está "entre 0 e 10".
nota_valida = df["quality"].between(0, 10)
print("Linhas removidas por quality inválida:", (~nota_valida).sum())
df = df[nota_valida].copy()   # o .copy() evita um aviso do pandas

# Confere se todas as notas são inteiras antes de converter
print("Notas não inteiras:", df["quality"].mod(1).ne(0).sum())
df["quality"] = df["quality"].astype(int)


# =========================================================
# 3. Coluna "alcohol"
# =========================================================
def corrigir_alcohol(texto):
    # Caso 1: o valor está vazio
    if pd.isna(texto):
        return np.nan

    # Caso 2: valor normal, como "9.4", "10" ou "11.35"
    if texto.count(".") <= 1:
        return float(texto)

    # Caso 3: o "11.8" repetido várias vezes
    if texto.startswith("11.811.8"):
        return 11.8

    # Caso 4: número com pontos de milhar, como "100.333.333.333.333"
    digitos = texto.replace(".", "")

    if digitos.startswith("1"):
        corrigido = digitos[:2] + "." + digitos[2:]   # entre 10 e 15
    else:
        corrigido = digitos[:1] + "." + digitos[1:]   # entre 8 e 9,99

    return float(corrigido)


df["alcohol"] = df["alcohol"].apply(corrigir_alcohol)

print(df["alcohol"].describe())
print("Valores vazios em alcohol:", df["alcohol"].isna().sum())


# =========================================================
# 4. Salva o resultado (sempre no final)
# =========================================================
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/winequality_clean.csv", index=False)