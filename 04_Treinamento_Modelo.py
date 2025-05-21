# Databricks notebook source
# MAGIC %md
# MAGIC ## Treinamento do Modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importação das Bibliotecas

# COMMAND ----------

#dbutils.library.restartPython()
from databricks import automl


# COMMAND ----------

# MAGIC %md
# MAGIC ### Salvando base de treinamento em um DF

# COMMAND ----------

df_treino = spark.read.table("_0903.int.cartao_fraude_treino_final")

# COMMAND ----------

df_treino.display()

# COMMAND ----------

from datetime import datetime

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

summary_normalized = automl.classify(
    dataset=df_treino,
    target_col="Class",
    timeout_minutes=20,
    experiment_name=f"analise_detec_fraude_{current_datetime}",
    primary_metric="recall",
    experiment_dir="/Workspace/Groups/databricks_coop_0903"
)

# Imprima o resumo dos resultados da classificação
print(summary_normalized)

# COMMAND ----------

# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
catalog = "_0903"
schema = "int"
model_name = "modelo_detec_fraude"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/5d50f24b823b4b84ac56d4954e6fcff2/model", f"{catalog}.{schema}.{model_name}")
