# Wine Quality Predictions

Trabalho final de Machine Learning: **predição e classificação da qualidade de vinhos** a partir de dados físico-químicos de vinhos tintos e brancos.

O objetivo é construir um pipeline completo de ML em duas partes:

1. **Regressão** — prever a nota de qualidade (`quality`, inteiro de 0 a 10) como valor contínuo.
2. **Classificação** — agrupar as notas em faixas de qualidade e prever a faixa.

## Estrutura do projeto

```
wine-quality-predictions/
├── data/
│   ├── raw/
│   │   └── group_3_winequality.csv   # dataset original fornecido pelo professor (grupo 3)
│   ├── processed/                    # saída do processing.py (ignorada pelo git)
│   └── processing.py                 # script de limpeza dos dados
└── notebooks/
    └── exploration.ipynb             # investigação dos problemas do dataset
```

## Como executar

Requisitos: Python 3 com `pandas` e `numpy` (e `jupyter` para o notebook).

```bash
pip install pandas numpy jupyter
```

Rode o script **a partir da raiz do repositório** (os caminhos são relativos a ela):

```bash
python data/processing.py
```

Isso gera `data/processed/winequality_clean.csv`. O notebook lê esse arquivo, então rode o script antes de abri-lo.

## Dataset

O dataset é uma versão do [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009) com erros inseridos de propósito. Ele tem 6.497 linhas e 13 colunas:

- 11 variáveis físico-químicas: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`
- `type`: `red` ou `white`
- `quality`: nota dada por degustadores (**target**)

## Limpeza dos dados (`data/processing.py`)

O script corrige os problemas encontrados no dataset bruto, coluna por coluna. A regra geral é: **quando dá para recuperar o valor certo, corrige; quando não dá, deixa vazio (`NaN`)**. Os vazios restantes serão preenchidos depois do split treino/teste, para não vazar informação do teste para o treino.

| Coluna | Problema | Tratamento |
|---|---|---|
| `quality` | Uma nota impossível (258) e uma vazia | Linhas removidas (o target não pode ser imputado); notas convertidas para `int` |
| `alcohol` | Texto corrompido: `"11.811.8..."` repetido e números com pontos de milhar (`"100.333.333..."`) | Lida como texto e reconstruída para a faixa 8–15 |
| `chlorides` | Quase todos os valores multiplicados por 1000 (0,076 → 76) | Valores ≥ 1 divididos por 1000; o que continua impossível vira vazio |
| `volatile acidity` | Parte dos valores multiplicada por 1000 (0,615 → 615) | Valores ≥ 50 divididos por 1000; valor ambíguo (4,94) vira vazio |
| `density` | Casas decimais perdidas (0,9978 → 9,978 / 99,78 / 997,8) e valores lidos como data pelo Excel (1,0014 → 1.2014) | Cada padrão é reconhecido pela faixa e devolvido à escala de ~0,99–1,04 |
| `type` | Palavra repetida (`"redredred..."`) | Reduzida para `red`/`white`; texto irreconhecível vira vazio |
| Outras 7 colunas | Valores isolados fora da faixa plausível | Limites com folga em relação às faixas do artigo original; fora deles vira vazio |

Por fim, as **duplicatas** são tratadas em duas etapas:

1. Remove linhas que têm um vazio numa coluna mas são idênticas a outra linha completa nas demais (a "gêmea" corrompida).
2. Remove linhas exatamente iguais, mantendo uma cópia.

### Resultado

| | Linhas |
|---|---|
| Dataset bruto | 6.497 |
| Após a limpeza | 5.317 (3.958 brancos, 1.359 tintos) |

Vazios que sobram para imputação: `sulphates` (95) e 1–2 em `fixed acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `density` e `pH`.

Distribuição do target após a limpeza:

| quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| vinhos | 30 | 206 | 1.752 | 2.320 | 856 | 148 | 5 |
