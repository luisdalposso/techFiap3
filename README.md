---
<details>
  <summary><strong>📖 Clique para expandir a explicação do notebook '01_get_dados'</strong></summary>

  Tem objetivo em fazer a coleta de dados utilizando a API do Kaggle, baixando o dataset de detecção de fraudes em cartão de crédito e carregá-lo em um Dataframe Spark e persisti-lo em uma tabela delta para consumo futuro.
  
### 🔧 1. Instalação de Pacotes
Instala bibliotecas essenciais para manipulação de dados, visualização, balanceamento de classes e machine learning com PySpark:

```bash
pip install pandas matplotlib imbalanced-learn pyspark
```

---

### 📦 2. Importação de Bibliotecas

Importa módulos como:

- `pandas`, `matplotlib.pyplot` — manipulação e visualização de dados.
- `SMOTE` — balanceamento de classes.
- `pyspark.ml` — criação de pipelines de machine learning.

---

### 🔐 3. Configuração da API do Kaggle

Cria e copia o arquivo `kaggle.json` com as credenciais de acesso à API do Kaggle para autenticação segura.

---

### 🔑 4. Autenticação com a API do Kaggle

Autentica o usuário com a API do Kaggle usando a biblioteca `KaggleApi`.

---

### 📥 5. Download do Dataset

Baixa o dataset `mlg-ulb/creditcardfraud` diretamente do Kaggle e salva localmente no diretório `/dbfs/FileStore/creditcard`.

---

### 📂 6. Leitura dos Dados com Spark

Carrega o arquivo CSV para um DataFrame do Spark com detecção automática de tipos de dados:

```python
df = spark.read.csv("file:/dbfs/FileStore/creditcard/creditcard.csv", header=True, inferSchema=True)
```

---

### 💾 7. Persistência em Tabela Delta

Salva o DataFrame como uma tabela Delta chamada `credit_card_fraud`, permitindo consultas SQL otimizadas:

```python
df.write.format("delta").mode("overwrite").saveAsTable("credit_card_fraud")
```

---

### 🔍 8. Consulta SQL

Exibe as primeiras linhas da tabela criada:

```sql
SELECT * FROM credit_card_fraud LIMIT 5
```

---
</details>

---
<details>
  <summary><strong>📖 Clique para expandir a explicação do notebook '02_EDA'</strong></summary>

Objetivo nessa etapa é explorar os dados de transações com cartão de crédito para entender o comportamento das fraudes. Usando PySpark, analisamos as estatísticas básicas, verifica se há dados null, observa padrões ao longo do tempo e compara as diferenças entre transações legítimas e fraudulentas. Também são gerados gráficos para facilitar a visualização e entender melhor como as variáveis se relacionam com a ocorrência de fraudes. Tudo isso ajuda a preparar o terreno para a próxima etapa: treinar modelos de machine learning.

### 🔧 1. Instalação de Pacotes
Instala bibliotecas essenciais para manipulação de dados, visualização, balanceamento de classes e machine learning com PySpark:

```bash
pip install pyspark matplotlib pandas seaborn
```

---

### 📦 2. Importação de Bibliotecas

Importa módulos como:

- `pandas`, `matplotlib`, `seaborn` — manipulação e visualização de dados.
- `pyspark` — manipulação de dados distribuídos

---

### 📂 3. Criação de nova coluna, Hour

Cria uma nova coluna nomeada 'Hour' com base no tempo em segundos a partir da primeira transação do dataset. 

```python
df = df.withColumn('Hour', floor(col('Time') / 3600).cast('int'))
```

---

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
