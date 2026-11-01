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
# 4. Coluna "chlorides"
# =========================================================
def corrigir_chlorides(valor):
    # Caso 1: o valor está vazio
    if pd.isna(valor):
        return np.nan

    # Caso 2: escala errada (0,076 virou 76)
    # Nenhum valor real chega a 1 g/L, então 1 ou mais está na escala errada
    if valor >= 1:
        valor = valor / 1000

    # Caso 3: mesmo depois da correção, continua impossível (o 1.920)
    # Não dá para saber o valor certo, então vira vazio
    # e será preenchido com a mediana depois do split
    if valor > 1:
        return np.nan

    return valor


print("Valores de chlorides na escala errada:", (df["chlorides"] >= 1).sum())
df["chlorides"] = df["chlorides"].apply(corrigir_chlorides)

print(df["chlorides"].describe())
print("Valores vazios em chlorides:", df["chlorides"].isna().sum())

# =========================================================
# 5. Coluna "volatile acidity"
# =========================================================
def corrigir_volatile_acidity(valor):
    # Caso 1: o valor está vazio
    if pd.isna(valor):
        return np.nan

    # Caso 2: escala certa (valores reais vão de cerca de 0,08 a 1,6)
    if valor < 2:
        return valor

    # Caso 3: escala errada (0,615 virou 615; 0,085 virou 85)
    if valor >= 50:
        return valor / 1000

    # Caso 4: entre 2 e 50, não se encaixa em nenhum grupo (o 4,94)
    # Vira vazio; a linha será removida na etapa de duplicatas,
    # porque existe uma linha gêmea com o valor correto (0,38)
    return np.nan


print("Valores de volatile acidity na escala errada:", (df["volatile acidity"] >= 50).sum())
df["volatile acidity"] = df["volatile acidity"].apply(corrigir_volatile_acidity)

print(df["volatile acidity"].describe())
print("Valores vazios em volatile acidity:", df["volatile acidity"].isna().sum())

# =========================================================
# 6. Coluna "density"
# =========================================================
def corrigir_density(valor):
    # Caso 1: o valor está vazio
    if pd.isna(valor):
        return np.nan

    # Caso 2: escala certa (valores reais vão de cerca de 0,987 a 1,039)
    if 0.98 <= valor <= 1.05:
        return valor

    # Caso 3: lido como data pelo Excel (1,0014 virou "1.2014", janeiro de 2014)
    if 1.19 <= valor <= 1.21:
        ano = round((valor - 1) * 10000)   # 1.2014 -> 2014
        return 1 + (ano % 100) / 10000     # 2014 -> 14 -> 1.0014

    # Caso 4: tinha 4 casas decimais (0,9978 virou 9,978)
    if 9.8 <= valor <= 10.5:
        return valor / 10

    # Caso 5: tinha 5 casas decimais (0,99704 virou 99,704)
    if 98 <= valor <= 105:
        return valor / 100

    # Caso 6: tinha 3 ou 6 casas decimais (0,997 virou 997)
    if 980 <= valor <= 1050:
        return valor / 1000

    # Caso 7: não se encaixa em nenhum grupo (o 436,832)
    # Vira vazio e será tratado depois
    return np.nan


df["density"] = df["density"].apply(corrigir_density)

print(df["density"].describe())
print("Valores vazios em density:", df["density"].isna().sum())

# =========================================================
# 7. Salva o resultado (sempre no final)
# =========================================================
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/winequality_clean.csv", index=False)