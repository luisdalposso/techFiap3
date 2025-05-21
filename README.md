# Tech Challenge - Detecção de Fraudes com Machine Learning no Databricks

## **Abstract**

Este projeto, desenvolvido como parte do Tech Challenge, visou a criação de um modelo de Machine Learning (ML) utilizando a plataforma **Databricks**, abrangendo todo o pipeline de dados, desde a coleta até o deployment do modelo em produção. O objetivo foi construir um sistema integrado e funcional que utiliza dados reais para treinar e servir um modelo preditivo, seguindo as seguintes etapas:

1. **Coleta e Armazenamento de Dados**:  
   - Dados de transações de cartão de crédito foram obtidos e armazenados em um **Data Lake** na estrutura do Databricks, utilizando tabelas Delta para garantir desempenho e escalabilidade.  
   - A coleta de dados foi automatizada utilizando a API do Kaggle.

2. **Exploração e Preparação dos Dados**:  
   - Realizada a **análise exploratória de dados (EDA)** para identificar padrões, lidar com dados ausentes, e entender a distribuição das classes (fraudes vs. não fraudes).  
   - Engenharia de features foi aplicada para remover variáveis redundantes e preparar os dados para o treinamento do modelo.

3. **Modelagem e Treinamento**:  
   - O modelo de ML foi treinado utilizando a biblioteca AutoML do Databricks, otimizando hiperparâmetros e avaliando diferentes algoritmos.  
   - A métrica principal utilizada foi o **recall**, priorizando a detecção de fraudes, uma vez que estas têm impacto financeiro significativo.

4. **Deployment**:  
   - O modelo treinado foi registrado e versionado no **MLflow**, integrando-o ao Unity Catalog do Databricks para gestão eficiente.  
   - Uma aplicação de deployment foi criada para alimentar um dashboard com previsões em tempo real.

5. **Visualização e Storytelling**:  
   - O storytelling do projeto foi apresentado em um vídeo explicativo, detalhando todas as etapas realizadas, desde a coleta de dados até o deployment do modelo.  
   - O vídeo inclui visualizações criadas a partir do Databricks e do modelo treinado.

## **Técnicas e Ferramentas Utilizadas**

### 1. **Coleta e Armazenamento**
   - Uso da **API do Kaggle** para baixar dados de transações de cartão de crédito.
   - Persistência de dados em tabelas Delta no Databricks para armazenamento otimizado.

### 2. **Análise e Preparação dos Dados**
   - **EDA (Exploratory Data Analysis)**: Estatísticas descritivas, análise de distribuições e correlação entre variáveis.
   - **Remoção de Features Irrelevantes**: Exclusão de colunas redundantes que não contribuem para a predição de fraudes.
   - **Criação de Features**: Criação de variáveis derivadas como "Hour" para capturar padrões temporais.

### 3. **Treinamento do Modelo**
   - **AutoML do Databricks**: Utilizado para selecionar automaticamente o melhor algoritmo e ajustar hiperparâmetros com base no **recall**.
   - **Pipeline de ML**: Construção de pipelines que incluem transformação de dados e avaliação do modelo.

### 4. **Deployment**
   - **MLflow**: Registro e versionamento do modelo no Unity Catalog do Databricks.
   - **Predições em Tempo Real**: Uso do modelo para gerar previsões e alimentar um dashboard com resultados.

### 5. **Visualização**
   - **Matplotlib e Seaborn**: Gráficos para visualização de distribuições e correlações.
   - **Storytelling em Vídeo**: Apresentação das etapas do projeto em formato visual.

## **Plataforma Utilizada**  
O projeto foi inteiramente desenvolvido no **Databricks**, uma plataforma unificada para engenharia de dados e aprendizado de máquina, que facilitou o armazenamento, processamento, treinamento e deployment do modelo de ML.

---

Este projeto cumpre os requisitos estabelecidos no desafio, apresentando um modelo funcional e documentado, integrado a um pipeline de dados robusto, e demonstrando sua aplicação prática por meio de um dashboard e um vídeo explicativo.


<details>
  <summary><strong>📖 Clique para expandir a explicação do notebook '01_get_dados'</strong></summary>

  Tem objetivo em fazer a coleta de dados utilizando a API do Kaggle, baixando o dataset de detecção de fraudes em cartão de crédito e carregá-lo em um Dataframe Spark e persisti-lo em uma tabela delta para consumo futuro.
  
### 🔧 1. Instalação de Pacotes
Instala bibliotecas essenciais para manipulação de dados, visualização, balanceamento de classes e machine learning com PySpark:

```bash
pip install pandas matplotlib imbalanced-learn pyspark
```



### 📦 2. Importação de Bibliotecas

Importa módulos como:

- `pandas`, `matplotlib.pyplot` — manipulação e visualização de dados.
- `SMOTE` — balanceamento de classes.
- `pyspark.ml` — criação de pipelines de machine learning.



### 🔐 3. Configuração da API do Kaggle

Cria e copia o arquivo `kaggle.json` com as credenciais de acesso à API do Kaggle para autenticação segura.



### 🔑 4. Autenticação com a API do Kaggle

Autentica o usuário com a API do Kaggle usando a biblioteca `KaggleApi`.



### 📥 5. Download do Dataset

Baixa o dataset `mlg-ulb/creditcardfraud` diretamente do Kaggle e salva localmente no diretório `/dbfs/FileStore/creditcard`.



### 📂 6. Leitura dos Dados com Spark

Carrega o arquivo CSV para um DataFrame do Spark com detecção automática de tipos de dados:

```python
df = spark.read.csv("file:/dbfs/FileStore/creditcard/creditcard.csv", header=True, inferSchema=True)
```



### 💾 7. Persistência em Tabela Delta

Salva o DataFrame como uma tabela Delta chamada `credit_card_fraud`, permitindo consultas SQL otimizadas:

```python
df.write.format("delta").mode("overwrite").saveAsTable("credit_card_fraud")
```



### 🔍 8. Consulta SQL

Exibe as primeiras linhas da tabela criada:

```sql
SELECT * FROM credit_card_fraud LIMIT 5
```
</details>


<details>
  <summary><strong>📖 Clique para expandir a explicação do notebook '02_EDA'</strong></summary>

Objetivo nessa etapa é explorar os dados de transações com cartão de crédito para entender o comportamento das fraudes. Usando PySpark, analisamos as estatísticas básicas, verifica se há dados null, observa padrões ao longo do tempo e compara as diferenças entre transações legítimas e fraudulentas. Também são gerados gráficos para facilitar a visualização e entender melhor como as variáveis se relacionam com a ocorrência de fraudes. Tudo isso ajuda a preparar o terreno para a próxima etapa: treinar modelos de machine learning.

### 🔧 1. Instalação de Pacotes
Instala bibliotecas essenciais para manipulação de dados, visualização, balanceamento de classes e machine learning com PySpark:

```bash
pip install pyspark matplotlib pandas seaborn
```



### 📦 2. Importação de Bibliotecas

Importa módulos como:

- `pandas`, `matplotlib`, `seaborn` — manipulação e visualização de dados.
- `pyspark` — manipulação de dados distribuídos



### 📂 3. Criação de nova coluna, Hour

Cria uma nova coluna nomeada 'Hour' com base no tempo em segundos a partir da primeira transação do dataset. 

```python
df = df.withColumn('Hour', floor(col('Time') / 3600).cast('int'))
```



### 🔎 4. Estátisticas descritivas

Geração de estatísticas descritivas para os atributos. Buscamos entender a distribuição dos dados, detectar possíveis outliers e verificar escalas diferentes entre as variáveis.

```python
df.describe('V1','V2','V3','V4','V5').show()
...
df.describe('V26','V27','V28', 'amount', 'time', 'hour').show()
```

### 🔎 5. Contagem de valores null

Verifica a existência de valores null nas colunas. Dados ausentes impactam em modelos de machine learning e precisam ser tratados adequadamente. No dataset em questão não houve valores null.

```python
null_counts = df.select([count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in df.columns])
null_counts.show()
```

### 🔦 6. Distribuição das Classes

Contagem de registros por classe, onde: 0 = não fraude; 1 = fraude. Buscamos aqui identificar desequilíbrio de classes. No dataset em questão observamos que ele era desbalanceado.

```python
df.groupBy("Class") \
    .count() \
    .withColumn("percentual", round((col("count") / total_contagem) * 100,2)) \
    .show()
```

### ♥️ 7. Matriz de Correlação

Cálculo da correlação entre as variáveis do dataset e visualização em formato headmap. Buscamos aqui identificar variaveis redudantes ou fortemente correlacionadas com a variável alvo (Class), úteis para o treinamento do modelo.

```python
corr = df_pandas.corr()
sns.heatmap(corr, ...)
```

### ♥️ 8. Correlação Individual com a Variável Alvo

Cálculo da correlação de cada variável com a variável CLASS. Ajudando assim a identificar quais variáveis têm maior relação com a ocorrência de fraudes.

```python
for col_name in df.columns:
    if col_name != "Class":
        print(f"{col_name}: {df.stat.corr(col_name, 'Class')}")
```

### 🔆 9. Distribuição de densidade por variável e classe

Plotagem da densidade de cada variável, separando por classe. Permite visualizar diferenças sutis na distribuição das variáveis entre classes, o que pode ser explorado por modelos de classificação.

```python
for idx, feature in enumerate(var):
    sns.kdeplot(t0[feature], ...)
    sns.kdeplot(t1[feature], ...)
```

</details>


<details> <summary><strong>📖 Clique para expandir a explicação do notebook '03_engenharia_features'</strong></summary>

Este notebook tem como objetivo realizar a engenharia de features no dataset de detecção de fraudes em cartões de crédito. Ele se concentra na preparação dos dados para a etapa de modelagem, removendo variáveis redundantes e persistindo a base final de treinamento.

📦 1. Importação de Bibliotecas

Importação de módulos para manipulação e transformação de dados:

```python
    pyspark.ml.feature.VectorAssembler — combinação de múltiplas colunas em um único vetor.
    pyspark.ml.stat.Summarizer — geração de estatísticas resumidas para os dados.
    pyspark.sql.functions — funções auxiliares para manipulação de DataFrames, como col, hour e when.
```

📂 2. Leitura da Tabela Delta

Carrega o dataset persistido na etapa anterior para um DataFrame Spark:

df = spark.read.table("_0903.int.cartao_fraude_treino")

🧹 3. Remoção de Colunas Irrelevantes

Identifica e remove colunas que possuem comportamento semelhante para as classes fraude e não fraude, baseando-se em análises exploratórias anteriores. Isso reduz a dimensionalidade do dataset e minimiza o impacto de variáveis redundantes nos modelos de machine learning.

Lista de colunas removidas:

    V13, V15, V22, V25, V26, V28

lista_remover_colunas = ['V13', 'V15', 'V22', 'V25', 'V26', 'V28']
df_modelo = df.drop(*lista_remover_colunas)

💾 4. Persistência da Base Final

Salva o DataFrame resultante como uma nova tabela Delta, garantindo que esteja otimizado para uso em modelos de machine learning:
```python
try:
    df.write.format("delta").mode("overwrite").saveAsTable("_0903.int.cartao_fraude_treino_final")
    print("\n7. Tabela '_0903.int.cartao_fraude_treino_final' criada com sucesso!")
except Exception as e:
    print(f"\n⚠️ AVISO: Não foi possível persistir ({str(e)})")
```

Essa tabela será usada como entrada na etapa de treinamento do modelo, contendo apenas as features mais relevantes.
</details>


<details>
  <summary><strong>📖 Clique para expandir a explicação do notebook '04_Treinamento_Modelo_MLflow'</strong></summary>

  O objetivo deste notebook é realizar o treinamento do modelo de detecção de fraudes usando ferramentas do Databricks, como o AutoML e o MLflow. Ele automatiza o processo de seleção de modelo e rastreia experimentos, permitindo versionamento e registro de modelos no catálogo do Databricks.

### 📦 1. Importação de Bibliotecas

Importa módulos essenciais para o treinamento e rastreamento do modelo:

- `databricks.automl` — biblioteca do Databricks para automatizar o treinamento e avaliação de modelos.
- `mlflow` — gerenciador de experimentos para rastrear e registrar modelos.



### 📂 2. Carregamento da Base de Treinamento

Carrega a base de dados preparada na etapa anterior como um DataFrame Spark:

```python
df_treino = spark.read.table("_0903.int.cartao_fraude_treino_final")
```

Isso garante que a base final, contendo apenas features relevantes, seja utilizada no treinamento.



### 🤖 3. Treinamento do Modelo com AutoML

Utiliza o AutoML para realizar as seguintes etapas automaticamente:
- Seleção de algoritmos de machine learning.
- Ajuste de hiperparâmetros.
- Avaliação baseada na métrica de recall.

**Parâmetros Configurados:**
- `target_col="Class"` — especifica a variável alvo.
- `timeout_minutes=20` — define um tempo máximo para a execução do AutoML.
- `primary_metric="recall"` — prioriza o recall, dado o foco em detectar fraudes.
- `experiment_name` — nomeia o experimento para rastreamento no MLflow.

```python
summary_normalized = automl.classify(
    dataset=df_treino,
    target_col="Class",
    timeout_minutes=20,
    experiment_name=f"analise_detec_fraude_{current_datetime}",
    primary_metric="recall",
    experiment_dir="/Workspace/Groups/databricks_0903"
)
```

### 📑 4. Registro do Modelo no MLflow

Após o treinamento, o modelo com melhor desempenho é registrado no catálogo do Databricks para versionamento e reutilização. O registro é feito diretamente a partir do run associado:

```python
mlflow.register_model("runs:/5d50f24b823b4b84ac56d4954e6fcff2/model", f"{catalog}.{schema}.{model_name}")
```

**Detalhes do Registro:**
- `runs:/...` — identifica o run do MLflow contendo o modelo treinado.
- `catalog`, `schema`, `model_name` — define o local onde o modelo será registrado no catálogo.

Este notebook facilita a automação do ciclo de vida do modelo, desde o treinamento até o rastreamento e registro, garantindo rastreabilidade e acessibilidade no Databricks.

</details>


<details>
  <summary><strong>📖 Clique para expandir a explicação do notebook '05_Modelagem_Deploy'</strong></summary>

  Este notebook foca no processo de deployment do modelo de detecção de fraudes no Databricks, incluindo carregamento do modelo treinado, geração de previsões e persistência dos resultados para consultas futuras.

### 📦 1. Configuração do Ambiente

- **Upgrade de bibliotecas**: Atualiza o MLflow para a versão mais recente e reinicia o ambiente Python para aplicar mudanças.

```bash
%pip install --upgrade "mlflow-skinny[databricks]"
dbutils.library.restartPython()
```

- **Configuração do Registro de Modelos**: Define o Unity Catalog como local para rastrear e gerenciar os modelos.

```python
mlflow.set_registry_uri("databricks-uc")
```

---

### 📂 2. Carregamento e Preparação do Modelo

Carrega o modelo registrado no MLflow para uso em previsões. A configuração do URI do modelo permite acesso ao modelo na fase de produção.

```python
model_uri = f"models:/{model_name}@{suffix}"
modelo_fraude = mlflow.sklearn.load_model(model_uri)
```

---

### 🧪 3. Processamento e Previsões

- **Seleção de Colunas**: Define um subconjunto de colunas relevantes para a entrada do modelo.
- **Geração de Previsões**: Utiliza o modelo carregado para prever probabilidades e classificar transações como fraudulentas ou não.

```python
y_deploy_proba = modelo_fraude.predict_proba(df_deploy_pandas)[:, 1]
y_deploy = (y_deploy_proba >= 0.90).astype('uint8')
```

---

### 💾 4. Persistência e Processamento de Resultados

- **Persistência de Resultados**: Salva os resultados das previsões como uma tabela Delta para uso em análises e relatórios futuros.
- **Filtro de Probabilidades**: Agrupa e visualiza os dados com base nas probabilidades preditas, permitindo análise detalhada das distribuições.

```python
if df_deploy.count() > 0:
    table_name = '_0903.exp.cartao_deploy_diaria'
    process_table(sdf_result, 'delta', datetime.now().strftime('%Y-%m-%d'))
```

- **Visualização**: Gera gráficos para analisar a distribuição de probabilidades.

```python
plt.bar(df_pandas['probabilidade'], df_pandas['count_high'], color='green', label='Probabilidade > 0.90')
plt.bar(df_pandas['probabilidade'], df_pandas['count_low'], color='red', label='Probabilidade <= 0.90')
```

---

Este notebook automatiza o pipeline de deployment do modelo, garantindo que as previsões sejam integradas eficientemente ao sistema de produção.

</details>
