
# %% [markdown]
# # Análise de Crédito com Interpretabilidade usando LIME

# %% [code]
# Instalação dos pacotes necessários (rode apenas uma vez)
!pip install pandas numpy scikit-learn lime matplotlib seaborn

# %% [code]
# Importações
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# %% [code]
# Carregar dados
colunas = [
    'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 'Savings', 'Employment',
    'InstallmentRate', 'PersonalStatusSex', 'Debtors', 'ResidenceDuration', 'Property',
    'Age', 'OtherInstallmentPlans', 'Housing', 'ExistingCredits', 'Job',
    'NumPeopleLiable', 'Telephone', 'ForeignWorker', 'Target'
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, sep=' ', header=None, names=colunas)

# Transformar Target: 1 = bom pagador, 2 = mau pagador → 1 = bom, 0 = mau
df['Target'] = df['Target'].map({1: 1, 2: 0})

# Visualizar dados
df.head()

# %% [code]
# Codificação de variáveis categóricas
df_encoded = df.copy()

for coluna in df_encoded.columns:
    if df_encoded[coluna].dtype == 'object':
        le = LabelEncoder()
        df_encoded[coluna] = le.fit_transform(df_encoded[coluna])

df_encoded.head()

# %% [code]
# Preparar dados para treino
X = df_encoded.drop('Target', axis=1)
y = df_encoded['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliar modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# %% [code]
# Configurar LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Mau pagador', 'Bom pagador'],
    mode='classification'
)

# %% [code]
# Função para gerar explicação textual clara
def gerar_explicacao_textual(explicacao, probas):
    favoravel = []
    desfavoravel = []

    for condicao, peso in explicacao.as_list():
        if peso > 0:
            favoravel.append(f"✅ {condicao} → contribuiu para 'bom pagador'.")
        else:
            desfavoravel.append(f"⚠️ {condicao} → aumenta risco de inadimplência.")

    texto = [f"**Análise detalhada:**"]
    texto.append(f"Probabilidades: Bom pagador = {probas[1]:.1%}, Mau pagador = {probas[0]:.1%}")

    if favoravel:
        texto.append("\n**Fatores positivos:**")
        texto.extend(favoravel)
    if desfavoravel:
        texto.append("\n**Fatores negativos:**")
        texto.extend(desfavoravel)

    texto.append("\n**Conclusão:**")
    if probas[1] > probas[0]:
        texto.append("🟢 O cliente tem **alta probabilidade de ser um bom pagador**.")
    else:
        texto.append("🔴 O cliente apresenta **risco de inadimplência**.")

    return "\n".join(texto)

# %% [code]
# Explicar cliente 5
i = 5
if i >= len(X_test):
    print(f"Erro: índice {i} é inválido. Tamanho do X_test: {len(X_test)}")
else:
    instancia = X_test.iloc[i]
    explicacao = explainer.explain_instance(
        data_row=instancia,
        predict_fn=modelo.predict_proba,
        num_features=10
    )

    probas = modelo.predict_proba([instancia])[0]
    texto_explicativo = gerar_explicacao_textual(explicacao, probas)

    print(f"\n🎯 Explicação para o cliente {i}:\n")
    print(texto_explicativo)

    # Mostrar gráfico
    fig = explicacao.as_pyplot_figure()
    plt.tight_layout()
    plt.show()

    # Salvar explicação
    explicacao.save_to_file('explicacao_cliente_5.html')

# %% [code]
# Explicar cliente 12
i = 12
if i >= len(X_test):
    print(f"Erro: índice {i} é inválido.")
else:
    instancia = X_test.iloc[i]
    explicacao = explainer.explain_instance(
        data_row=instancia,
        predict_fn=modelo.predict_proba,
        num_features=10
    )

    probas = modelo.predict_proba([instancia])[0]
    texto_explicativo = gerar_explicacao_textual(explicacao, probas)

    print(f"\n🎯 Explicação para o cliente {i}:\n")
    print(texto_explicativo)

    fig = explicacao.as_pyplot_figure()
    plt.tight_layout()
    plt.show()

    explicacao.save_to_file('explicacao_cliente_12.html')

# %% [code]
# Conclusão
print("""
💡 Conclusão:
- A técnica LIME trouxe transparência ao modelo de classificação de crédito.
- Fatores como histórico de crédito, emprego e idade foram decisivos.
- A interpretabilidade é essencial para confiança, justiça e conformidade regulatória em sistemas de decisão automatizados.
""")