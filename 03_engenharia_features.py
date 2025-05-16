# Databricks notebook source
# MAGIC %md
# MAGIC # Engenharia de Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importar bibliotecas

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Summarizer
from pyspark.sql.functions import col, hour, when



# COMMAND ----------

df = spark.read.table("credit_card_fraud")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remover as colunas que possuem comportamento semelhantes para a classe fraude e não fraude

# COMMAND ----------

lista_remover_colunas = ['V13', 'V15', 'V22', 'V25', 'V26', 'V28']

# COMMAND ----------

df_modelo = df.drop(*lista_remover_colunas)
df_modelo.display()

# COMMAND ----------


try:
    df_modelo.write.format("delta").mode("overwrite").saveAsTable("credit_card_fraud_modelo")
    print("\n7. Tabela 'credit_card_fraud_modelo' criada com sucesso!")
except Exception as e:
    print(f"\n⚠️ AVISO: Não foi possível persistir ({str(e)})")