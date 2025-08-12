# Explicabilidade de Modelos Preditivos de Crédito com LIME

Modelos de machine learning são amplamente utilizados em sistemas de análise de crédito bancário, uma vez que conseguem prever com alta precisão se um cliente é um bom ou mau pagador. No entanto, esses modelos são frequentemente criticados por atuarem como "caixas-pretas".

Em setores altamente regulados como o financeiro, é essencial não apenas decidir corretamente, mas também ser capaz de justificar cada decisão. Assim, o objetivo deste projeto é aplicar técnicas de Explainable AI (XAI), utilizando a biblioteca LIME, para tornar as previsões de um modelo preditivo mais interpretáveis e transparentes.

# Modelo Preditivo Escolhido

Foi utilizado o modelo Random Forest, uma técnica de ensemble que combina várias árvores de decisão para obter resultados mais robustos e menos propensos a overfitting. A escolha do Random Forest se deu por sua capacidade de lidar com variáveis categóricas e numéricas, bem como por sua boa performance em tarefas de classificação binária.

# Etapas:

Carregamento e nomeação do dataset "Statlog (German Credit Data)" da UCI.

Codificação de variáveis categóricas.

Divisão entre treino (80%) e teste (20%).

Treinamento do modelo RandomForestClassifier.

Avaliação com classification_report (acurácia ~81%).

# Explicabilidade com LIME

A biblioteca LIME (Local Interpretable Model-agnostic Explanations) foi utilizada para gerar explicações locais sobre as previsões do modelo. Com LIME, é possível identificar quais variáveis mais influenciaram a decisão de classificar um cliente como "bom pagador" ou "mau pagador".

Exemplos de saída:

Gráficos com as features mais relevantes para cada previsão.

Análise textual com impacto positivo ou negativo de cada variável.

Essas explicações permitem que analistas de crédito, clientes e órgãos reguladores compreendam a decisão do modelo, aumentando a confiança no sistema automatizado.

# Reflexões e Limitações

Apesar das vantagens, a interpretabilidade local fornecida pelo LIME pode não refletir toda a complexidade do modelo global. Além disso, escolhas como o número de vizinhos ou a granularidade da discretização influenciam diretamente a explicação.

Ainda assim, o uso do LIME representa um avanço significativo rumo a modelos mais transparentes e auditáveis. Ao observar os fatores que pesaram contra ou a favor de um cliente, é possível também orientar melhorias nos dados ou no processo de decisão humana.

Com o avanço da Inteligência Artificial, modelos preditivos têm sido amplamente utilizados em decisões sensíveis, como a concessão de crédito. Embora precisos, esses modelos frequentemente funcionam como uma “caixa-preta”, dificultando a compreensão de suas decisões por parte de clientes, gerentes e órgãos regulatórios.

Este projeto utiliza a técnica de **XAI (Explainable Artificial Intelligence)** com a biblioteca **LIME**, para tornar transparentes as decisões de um modelo que classifica clientes como “bom” ou “mau” pagador.

---

# Objetivos

- Treinar um modelo de classificação para risco de crédito.
- Aplicar o LIME para explicar as decisões do modelo de forma individualizada.
- Promover a transparência do modelo preditivo e facilitar sua interpretação por stakeholders.

---

#  Modelo Preditivo

Utilizamos o modelo **Random Forest Classifier**, uma técnica de aprendizado de máquina baseada em múltiplas árvores de decisão. Ele foi escolhido por oferecer bom desempenho e estabilidade em classificações binárias como essa.

- **Dados:** Statlog (German Credit Data) — UCI Repository
- **Variáveis:** 20 atributos como idade, conta bancária, histórico de crédito, renda, etc.
- **Classes:** 1 (bom pagador), 0 (mau pagador)
- **Acurácia obtida:** ~81%

---

# Técnicas de Explicabilidade

O **LIME (Local Interpretable Model-agnostic Explanations)** foi usado para gerar explicações locais — ou seja, explicar cada decisão do modelo individualmente. Isso é feito analisando como pequenas mudanças nas variáveis influenciam a saída do modelo.

Exemplo de explicação para um cliente:

- `CreditHistory > 2` ➜ Aumentou chance de aprovação
- `Status <= 3` ➜ Diminuiu chance de aprovação

Gráficos foram gerados para mostrar visualmente as variáveis mais relevantes para cada cliente analisado.

---

# Outputs

Os principais outputs do projeto são:

- 📊 **Gráfico com explicações geradas pelo LIME**  
- 📋 **Tabela textual com os impactos das variáveis**
- 🧾 **Relatório de classificação** com métricas (precision, recall, f1-score)

 Veja a pasta `/imagens` com alguns exemplos.

---

# Limitações e Reflexões

Embora o LIME ajude a entender as decisões do modelo, ele possui limitações:

- Pode gerar explicações inconsistentes se o modelo for muito não linear.
- Explicações locais não representam o comportamento global do modelo.
- A escolha dos dados vizinhos (perturbações) pode afetar os resultados.

No entanto, a interpretabilidade:
-  *Promove confiança*
- *Atende exigências regulatórias*
- *Ajuda equipes a entender e melhorar o modelo*

---

# Execução do Projeto

*1. Clone o repositório:*

git clone https://github.com/seu-usuario/explicabilidade-modelo-credito.git
cd explicabilidade-modelo-credito

*2. Instale as dependências:*


pip install -r requirements.txt

*3. execute o notebook:*

Abra o projeto_lime.ipynb no Jupyter ou Google Colab.


*4. Dataset utilizado:*

German Credit Data - UCI Machine Learning Repository

**Lembrando que, como o o projeto foi feito a partir do Google Colab, não seria necessario baixar o arquivo manualmente, pois podemos pegar diretamente do link.**
