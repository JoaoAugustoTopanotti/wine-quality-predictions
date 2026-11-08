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
├── src/
│   └── preparacao.py                 # split treino/teste e pipeline de pré-processamento
└── notebooks/
    ├── raw_data_investigation.ipynb  # investigação dos problemas do dataset bruto
    ├── exploratory_analisys.ipynb    # análise exploratória do dataset limpo
    ├── regression.ipynb              # Parte 1: modelos de regressão
    ├── classification.ipynb          # Parte 2: modelos de classificação
    └── demonstracao.ipynb            # testes dos modelos finais com vinhos novos
```

## Como executar

Requisitos: Python 3 com `pandas`, `numpy`, `matplotlib`, `seaborn` (seaborn 0.13 ou superior), `scikit-learn`, `imbalanced-learn` e `jupyter`.

```bash
pip install pandas numpy matplotlib "seaborn>=0.13" scikit-learn imbalanced-learn jupyter
```

Rode o script **a partir da raiz do repositório** (os caminhos são relativos a ela):

```bash
python data/processing.py
```

Isso gera `data/processed/winequality_clean.csv`. Os notebooks leem esse arquivo, então rode o script antes de abri-los.

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

## Análise exploratória (`notebooks/exploratory_analisys.ipynb`)

O notebook explora o dataset limpo em seções: visão geral, distribuição do target,
histogramas, outliers (boxplots), relação das variáveis com a `quality`, scatter plots e
matriz de correlação. Os tintos e brancos são analisados separadamente em todas as etapas.

Principais conclusões:

- **`alcohol` é a variável mais relacionada à qualidade** (correlação ≈ 0,47, igual nos dois tipos). Nenhuma outra passa de 0,40, então um R² modesto é esperado na regressão.
- **As variáveis afetam a nota de forma diferente em cada tipo.** `volatile acidity`, `sulphates` e `citric acid` importam mais nos tintos; `chlorides` e `density`, nos brancos. Por isso o `type` entra como variável do modelo.
- **O target é desbalanceado:** as notas 5 e 6 somam cerca de 77% dos vinhos. O split treino/teste será estratificado pela `quality`.
- **Multicolinearidade** em três pares: `alcohol` × `density`, `residual sugar` × `density` e `free` × `total sulfur dioxide`.
- **Os outliers são valores reais** e serão mantidos.

As decisões completas de pré-processamento e modelagem estão na seção 9 do notebook.

## Regressão (`notebooks/regression.ipynb`)

Parte 1: prever a `quality` como valor contínuo.

**Preparação** (`src/preparacao.py`, compartilhado com a Parte 2): split 80/20 estratificado pela `quality` e um `Pipeline` do scikit-learn que preenche os vazios com a mediana, codifica o `type` como 0/1 e padroniza as variáveis. Como tudo fica dentro do pipeline, essas etapas são ajustadas só no treino, inclusive dentro de cada rodada da validação cruzada.

**Modelos:** Ridge, Random Forest e Gradient Boosting (`HistGradientBoostingRegressor`), comparados com um baseline que chuta sempre a média. Os hiperparâmetros foram ajustados com `GridSearchCV` (validação cruzada de 5 folds, otimizando o RMSE).

Resultados no conjunto de teste:

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (média) | 0,880 | 0,695 | 0,000 |
| Ridge | 0,721 | 0,557 | 0,329 |
| **Random Forest** | **0,679** | **0,521** | **0,406** |
| Gradient Boosting | 0,684 | 0,527 | 0,397 |

Principais conclusões:

- **O Random Forest foi o melhor modelo**, errando a nota em cerca de meio ponto. O Gradient Boosting empatou na prática.
- **O ajuste de hiperparâmetros trouxe pouco ganho** (menos de 0,01 de RMSE). No Ridge, o que mais ajudou foi o log nas variáveis de cauda longa.
- **As variáveis mais importantes são `alcohol`, `volatile acidity` e `free sulfur dioxide`**, nos dois modelos e de acordo com a EDA.
- **Os modelos erram muito nas notas extremas** (3, 4, 8 e 9), puxando as previsões para o centro. Isso motiva a Parte 2, de classificação em faixas.

## Classificação (`notebooks/classification.ipynb`)

Parte 2: agrupar as notas em faixas de qualidade e prever a faixa. Usa o mesmo split e o mesmo pré-processamento da regressão.

**Faixas:** três esquemas foram comparados. O principal é o **A**, por ser o mais informativo:

| Esquema | Classes | Proporção | F1 macro (Random Forest, validação cruzada) |
|---|---|---|---|
| **A** | baixa (3–4) / média (5–6) / alta (7–9) | 4% / 77% / 19% | 0,55 |
| B | ruim (≤ 5) / bom (≥ 6) | 37% / 63% | 0,75 |
| C | comum (≤ 6) / excelente (≥ 7) | 81% / 19% | 0,73 |

**Métrica principal:** F1 macro, que dá o mesmo peso a todas as classes. A acurácia engana com classes desbalanceadas: no esquema A, responder sempre "média" já acerta 77%.

**Balanceamento:** `class_weight="balanced"` e SMOTE tiveram resultados equivalentes. O `class_weight` foi escolhido por ser mais simples.

**Modelos:** Regressão Logística, Random Forest e Gradient Boosting (`HistGradientBoostingClassifier`), ajustados com `GridSearchCV`.

Resultados no conjunto de teste (esquema A):

| Modelo | F1 macro | Balanced acc. | Acurácia |
|---|---|---|---|
| Baseline (classe mais comum) | 0,289 | 0,333 | 0,766 |
| Regressão Logística | 0,477 | 0,685 | 0,558 |
| **Random Forest** | **0,603** | **0,658** | **0,749** |
| Gradient Boosting | 0,589 | 0,634 | 0,741 |

Principais conclusões:

- **O Random Forest foi o melhor modelo**; o Gradient Boosting empatou na prática. A Regressão Logística encontra mais vinhos das classes raras, mas com muitos alarmes falsos.
- **As confusões acontecem entre faixas vizinhas** e se concentram nas notas de fronteira (nota 6 prevista como "alta", nota 4 prevista como "média"). A classe baixa é a mais difícil (F1 de 0,39).
- **As variáveis decisivas são `alcohol`, `free sulfur dioxide` e `volatile acidity`**, as mesmas da regressão.
- Com o balanceamento, o modelo identifica 71% dos vinhos de qualidade alta e cerca de metade dos de qualidade baixa, algo que a regressão não conseguia.

## Demonstração (`notebooks/demonstracao.ipynb`)

Um "laboratório" para usar os modelos finais (Random Forest de regressão e de classificação). Eles são treinados de novo com os melhores hiperparâmetros das Partes 1 e 2 e saem idênticos aos originais (mesmas métricas no teste). O notebook:

- **classifica qualquer vinho** a partir das 12 variáveis, com a nota prevista, a faixa e a probabilidade de cada faixa (basta editar os valores na seção 4.2);
- **compara previsão e realidade** em um vinho sorteado de cada nota do teste;
- **simula o efeito de cada variável** ("e se?"): partindo de um vinho típico, varia o `alcohol`, a `volatile acidity` e o `free sulfur dioxide`, um de cada vez.

As simulações mostram o que o modelo aprendeu (associações), não como fazer um vinho melhor: mudar uma variável sozinha pode gerar combinações que não existem na prática.
