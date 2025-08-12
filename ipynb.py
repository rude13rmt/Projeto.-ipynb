{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MjEKpK3u0cey"
      },
      "outputs": [],
      "source": [
        "!pip install pandas numpy scikit-learn lime matplotlib seaborn\n",
        "!pip install lime"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Nomes das colunas\n",
        "colunas = [\n",
        "    'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 'Savings', 'Employment',\n",
        "    'InstallmentRate', 'PersonalStatusSex', 'Debtors', 'ResidenceDuration', 'Property',\n",
        "    'Age', 'OtherInstallmentPlans', 'Housing', 'ExistingCredits', 'Job',\n",
        "    'NumPeopleLiable', 'Telephone', 'ForeignWorker', 'Target'\n",
        "]\n",
        "\n",
        "# URL do dataset\n",
        "url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data\"\n",
        "\n",
        "# Lê os dados direto da internet\n",
        "df = pd.read_csv(url, sep=' ', header=None, names=colunas)\n",
        "\n",
        "# Altera a coluna \"Target\": 1 = bom pagador, 2 = mau pagador\n",
        "df['Target'] = df['Target'].map({1: 1, 2: 0})\n",
        "\n",
        "# Visualiza os 5 primeiros registros\n",
        "df.head()\n"
      ],
      "metadata": {
        "id": "Ba2ziC382koe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "\n",
        "# Faz uma cópia do DataFrame original para não mexer nos dados crus\n",
        "df_encoded = df.copy()\n",
        "\n",
        "# Converte colunas categóricas para números\n",
        "for coluna in df_encoded.columns:\n",
        "    if df_encoded[coluna].dtype == 'object':\n",
        "        le = LabelEncoder()\n",
        "        df_encoded[coluna] = le.fit_transform(df_encoded[coluna])\n"
      ],
      "metadata": {
        "id": "ekSN30Ww25u0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_encoded.head()\n"
      ],
      "metadata": {
        "id": "WM0JYJiM29rU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import classification_report\n",
        "\n",
        "# Separa os dados em X (variáveis) e y (resposta)\n",
        "X = df_encoded.drop('Target', axis=1)\n",
        "y = df_encoded['Target']\n",
        "\n",
        "# Divide em treino (80%) e teste (20%)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Cria e treina o modelo\n",
        "modelo = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "modelo.fit(X_train, y_train)\n",
        "\n",
        "# Faz previsões\n",
        "y_pred = modelo.predict(X_test)\n",
        "\n",
        "# Mostra o relatório de desempenho\n",
        "print(classification_report(y_test, y_pred))\n"
      ],
      "metadata": {
        "id": "y7mUcnMh3O3U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import lime\n",
        "import lime.lime_tabular\n",
        "import numpy as np\n",
        "\n",
        "explainer = lime.lime_tabular.LimeTabularExplainer(\n",
        "    training_data=np.array(X_train),\n",
        "    feature_names=X.columns,\n",
        "    class_names=['Mau pagador', 'Bom pagador'],\n",
        "    mode='classification'\n",
        ")\n"
      ],
      "metadata": {
        "id": "9mg2K8bRAijW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Escolhe uma pessoa (ex: pessoa número 5 no conjunto de teste)\n",
        "i = 5\n",
        "\n",
        "# Gera a explicação\n",
        "explicacao = explainer.explain_instance(\n",
        "    data_row=X_test.iloc[i],\n",
        "    predict_fn=modelo.predict_proba\n",
        ")\n",
        "\n",
        "# Mostra no notebook\n",
        "explicacao.show_in_notebook(show_all=False)\n"
      ],
      "metadata": {
        "id": "8wvoB-V8Ajtu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Função que gera explicação textual\n",
        "def gerar_explicacao_textual(explicacao, dados_instancia, probas):\n",
        "    conclusao = []\n",
        "    favoravel = []\n",
        "    desfavoravel = []\n",
        "\n",
        "    for condicao, peso in explicacao.as_list():\n",
        "        texto_base = \"\"\n",
        "\n",
        "        if peso > 0:\n",
        "            texto_base += f\"{condicao} contribuiu positivamente para a classificação como bom pagador.\"\n",
        "            favoravel.append(texto_base)\n",
        "        else:\n",
        "            texto_base += f\"{condicao} contribuiu negativamente, indicando risco de inadimplência.\"\n",
        "            desfavoravel.append(texto_base)\n",
        "\n",
        "    conclusao.append(f\"**Resumo da análise da instância:**\\n\")\n",
        "    conclusao.append(f\"O modelo atribuiu uma probabilidade de **{probas[1]:.1%}** para que a pessoa seja um **bom pagador**, e **{probas[0]:.1%}** de ser um **mau pagador**.\\n\")\n",
        "\n",
        "    if favoravel:\n",
        "        conclusao.append(\"\\n**Fatores que influenciaram positivamente:**\")\n",
        "        conclusao.extend(favoravel)\n",
        "\n",
        "    if desfavoravel:\n",
        "        conclusao.append(\"\\n**Fatores que influenciaram negativamente:**\")\n",
        "        conclusao.extend(desfavoravel)\n",
        "\n",
        "    conclusao.append(\"\\n**Conclusão:**\")\n",
        "\n",
        "    if probas[1] > probas[0]:\n",
        "        conclusao.append(\"O modelo, com base nos fatores acima, considera que a pessoa tem maior chance de ser **bom pagador**.\")\n",
        "    else:\n",
        "        conclusao.append(\"O modelo, com base nos fatores acima, indica maior risco de **inadimplência (mau pagador)**.\")\n",
        "\n",
        "    return \"\\n\".join(conclusao)\n",
        "\n",
        "\n",
        "# -----------------------------\n",
        "# Seleciona a instância (exemplo: pessoa número 5)\n",
        "i = 5\n",
        "\n",
        "# Gera a explicação com LIME\n",
        "explicacao = explainer.explain_instance(\n",
        "    data_row=X_test.iloc[i],\n",
        "    predict_fn=modelo.predict_proba\n",
        ")\n",
        "\n",
        "# Calcula as probabilidades da predição\n",
        "probas = modelo.predict_proba([X_test.iloc[i]])[0]\n",
        "\n",
        "# Recupera os dados da pessoa analisada\n",
        "dados_instancia = X_test.iloc[i]\n",
        "\n",
        "# Gera o texto explicativo\n",
        "texto_explicativo = gerar_explicacao_textual(explicacao, dados_instancia, probas)\n",
        "\n",
        "# Exibe no terminal/notebook\n",
        "print(texto_explicativo)\n",
        "\n",
        "print(texto_explicativo)\n"
      ],
      "metadata": {
        "id": "Scuc-NEEPz9r"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "explicacao.save_to_file('explicacao_lime.html')\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "D8hGRWVYB_oX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Escolhe outro cliente (ex: cliente número 12 do conjunto de teste)\n",
        "i = 12\n",
        "\n",
        "# Gera a explicação\n",
        "explicacao = explainer.explain_instance(\n",
        "    data_row=X_test.iloc[i],\n",
        "    predict_fn=modelo.predict_proba\n",
        ")\n",
        "\n",
        "# Mostra explicação em gráfico limpo\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "fig = explicacao.as_pyplot_figure()\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Mostra explicação em texto limpo\n",
        "print(f\"\\nExplicação para o cliente {i}:\\n\")\n",
        "for feature, weight in explicacao.as_list():\n",
        "    direcao = \"↑\" if weight > 0 else \"↓\"\n",
        "    print(f\"{direcao} {feature} → impacto de {weight:.3f}\")\n",
        "\n",
        "# Probabilidade prevista\n",
        "proba = modelo.predict_proba([X_test.iloc[i]])[0]\n",
        "classe_prevista = modelo.predict([X_test.iloc[i]])[0]\n",
        "print(f\"\\nModelo prevê: {'Bom pagador' if classe_prevista == 1 else 'Mau pagador'}\")\n",
        "print(f\"Probabilidades: Mau = {proba[0]:.2f} | Bom = {proba[1]:.2f}\")\n"
      ],
      "metadata": {
        "id": "HUFhYAmvDWgG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Função explicativa com base nos pesos\n",
        "def gerar_explicacao_textual(explicacao, dados_instancia, probas):\n",
        "    conclusao = []\n",
        "    favoravel = []\n",
        "    desfavoravel = []\n",
        "\n",
        "    for condicao, peso in explicacao.as_list():\n",
        "        texto_base = \"\"\n",
        "\n",
        "        if peso > 0:\n",
        "            texto_base += f\"A condição '{condicao}' teve um impacto positivo na predição, favorecendo a classificação como bom pagador.\"\n",
        "            favoravel.append(texto_base)\n",
        "        else:\n",
        "            texto_base += f\"A condição '{condicao}' teve um impacto negativo, sugerindo maior risco de inadimplência.\"\n",
        "            desfavoravel.append(texto_base)\n",
        "\n",
        "    conclusao.append(f\"**Resumo da análise do cliente:**\\n\")\n",
        "    conclusao.append(f\"O modelo atribuiu uma probabilidade de **{probas[1]:.1%}** para ser **bom pagador**, e **{probas[0]:.1%}** para ser **mau pagador**.\\n\")\n",
        "\n",
        "    if favoravel:\n",
        "        conclusao.append(\"**Fatores positivos identificados:**\")\n",
        "        conclusao.extend(favoravel)\n",
        "\n",
        "    if desfavoravel:\n",
        "        conclusao.append(\"\\n**Fatores negativos identificados:**\")\n",
        "        conclusao.extend(desfavoravel)\n",
        "\n",
        "    conclusao.append(\"\\n**Conclusão:**\")\n",
        "    if probas[1] > probas[0]:\n",
        "        conclusao.append(\"Com base nos fatores acima, o modelo indica maior probabilidade de bom pagamento.\")\n",
        "    else:\n",
        "        conclusao.append(\"Com base nos fatores acima, o modelo sinaliza risco de inadimplência.\")\n",
        "\n",
        "    return \"\\n\".join(conclusao)\n",
        "\n",
        "\n",
        "# Cliente escolhido\n",
        "i = 12\n",
        "\n",
        "# Gera explicação LIME\n",
        "explicacao = explainer.explain_instance(\n",
        "    data_row=X_test.iloc[i],\n",
        "    predict_fn=modelo.predict_proba\n",
        ")\n",
        "\n",
        "# Gráfico LIME\n",
        "import matplotlib.pyplot as plt\n",
        "fig = explicacao.as_pyplot_figure()\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Calcula probabilidade e dados\n",
        "probas = modelo.predict_proba([X_test.iloc[i]])[0]\n",
        "dados_instancia = X_test.iloc[i]\n",
        "\n",
        "# Gera e exibe explicação textual\n",
        "texto_explicativo = gerar_explicacao_textual(explicacao, dados_instancia, probas)\n",
        "print(f\"\\nExplicação para o cliente {i}:\\n\")\n",
        "print(texto_explicativo)\n"
      ],
      "metadata": {
        "id": "ngxTd67ZRJjC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"\"\"\n",
        "A aplicação da técnica LIME permitiu entender claramente como o modelo de crédito toma decisões.\n",
        "Variáveis como histórico de crédito, renda e tempo de trabalho foram essenciais para classificar clientes.\n",
        "\n",
        "A interpretabilidade ajuda a promover transparência, confiança e aderência regulatória — especialmente em setores como o bancário.\n",
        "\"\"\")\n"
      ],
      "metadata": {
        "id": "L0i6YD0uEG-P"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}