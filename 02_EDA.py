# Databricks notebook source
# MAGIC %md
# MAGIC # EDA - Análise Exploratória de Dados

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importar bibliotecas que serão usadas

# COMMAND ----------


from pyspark.sql.functions import col, isnan, when, count, round, mean, stddev, abs, floor
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# COMMAND ----------

# MAGIC %md
# MAGIC ### Guardar base do Catalog em um Dataframe

# COMMAND ----------

# guardar num dataframe o conteudo da tabela delta
df = spark.read.table("credit_card_fraud")
display(df.limit(5))

# COMMAND ----------


df = df.withColumn('Hour', floor(col('Time') / 3600).cast('int'))
df.display()



# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificar as Colunas e Tipagem dos dados

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Class = 0 - Transação Válida
# MAGIC #### Class = 1 - Fraude

# COMMAND ----------

# MAGIC %md
# MAGIC ### Descrição Geral do Dataset

# COMMAND ----------

df.describe('V1','V2','V3','V4','V5').show()

# COMMAND ----------

df.describe('V6','V7','V8','V9','V10').show()

# COMMAND ----------

df.describe('V11','V12','V13','V14','V15').show()

# COMMAND ----------

df.describe('V16','V17','V18','V19','V20').show()

# COMMAND ----------

df.describe('V21','V22','V23','V24','V25').show()

# COMMAND ----------

df.describe('V26','V27','V28', 'amount', 'time', 'hour').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificar valores 'null' nas colunas

# COMMAND ----------

# Contar valores nulos em cada coluna
null_counts = df.select([count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in df.columns])
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Não existem dados nulos no dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificação dos Balanceamento das Classes (1 = Fraude e 0 = Válida)

# COMMAND ----------

# Total de registro

total_contagem = df.count()

# Contagem das classes + percentual

df.groupBy("Class") \
    .count() \
    .withColumn("percentual", round((col("count") / total_contagem) * 100,2)) \
    .show()

# COMMAND ----------

# Total de registro
total_contagem = df.count()

# Contagem das classes
class_counts = df.groupBy("Class") \
    .count() \
    .toPandas()

# Criar gráfico de barras
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='Class', y='count', data=class_counts)
plt.xlabel('Class')
plt.ylabel('Contagem')
plt.title('Distribuição das Classes')

# Adicionar rótulos aos dados
for index, row in class_counts.iterrows():
    barplot.text(row.name, row['count'], row['count'], color='black', ha="center")

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### A nossa classe (class) esta muito desbalanceada

# COMMAND ----------


# Filtrar transações fraudulentas e não fraudulentas
fraudes = df.filter(col('Class') == 1).select('Time').toPandas()
validas = df.filter(col('Class') == 0).select('Time').toPandas()

# Configurar o tamanho da figura
plt.figure(figsize=(10, 4))

# Criar histogramas para transações não fraudulentas e fraudulentas
sns.histplot(validas['Time'], bins=50, color='gray', label='Não Fraudulentas', kde=False, stat="count")
sns.histplot(fraudes['Time'], bins=50, color='red', label='Fraudulentas', kde=False, stat="count")

# Adicionar título e rótulos aos eixos
plt.title('Quantidade de Transações ao Longo do Tempo')
plt.xlabel('Time (segundos)')
plt.ylabel('Quantidade de Transações')
plt.legend()

# Mostrar o gráfico
plt.show()


# COMMAND ----------

# Plotar a distribuição das transações ao longo do tempo
plt.figure(figsize=(10, 4))
sns.histplot(validas['Time'], bins=50, color='gray', label='Não Fraudulentas', kde=True, stat="density", common_norm=False)
sns.histplot(fraudes['Time'], bins=50, color='red', label='Fraudulentas', kde=True, stat="density", common_norm=False)
plt.title('Distribuição das Transações ao Longo do Tempo')
plt.xlabel('Time (segundos)')
plt.ylabel('Densidade')
plt.legend()

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Detecção de Outliers

# COMMAND ----------


# Converter para Pandas DataFrame para visualização
df_pandas = df.toPandas()

# Plotar os gráficos de boxplot com e sem outliers
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

# Boxplot com outliers
sns.boxplot(ax=ax1, x="Class", y="Amount", hue="Class", data=df_pandas, palette="Paired", showfliers=True)
ax1.set_title("Com Outliers")

# Boxplot sem outliers
sns.boxplot(ax=ax2, x="Class", y="Amount", hue="Class", data=df_pandas, palette="Paired", showfliers=False)
ax2.set_title("Sem Outliers")

plt.show()




# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificação da Correlação das Variáveis com Target (Class)

# COMMAND ----------


# Calcular a correlação
corr = df_pandas.corr()

# Plotar o heatmap de correlação
plt.figure(figsize=(18, 8))
plt.title('Correlação entre as variáveis (Pearson)')
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap=sns.cubehelix_palette(as_cmap=True), annot=True, fmt=".1f")
plt.show()


# COMMAND ----------

for col_name in df.columns:
    if col_name != "Class":
        print(f"{col_name}: {df.stat.corr(col_name, 'Class')}")

# COMMAND ----------


# Densidade de todas as variáveis, comparando as duas classes
var = df_pandas.columns.values

t0 = df_pandas[df_pandas['Class'] == 0]
t1 = df_pandas[df_pandas['Class'] == 1]

plt.figure()
fig, ax = plt.subplots(8, 4, figsize=(16, 28))

for idx, feature in enumerate(var):
    plt.subplot(8, 4, idx + 1)
    sns.kdeplot(t0[feature], bw_method='scott', bw_adjust=0.5, label="Class = 0", color='gray')
    sns.kdeplot(t1[feature], bw_method='scott', bw_adjust=0.5, label="Class = 1", color='red')
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### classes não separadas = V13, V15, V22, V25, V26 e V28