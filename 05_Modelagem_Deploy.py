# Databricks notebook source
# DBTITLE 1,Upgrade mlflow e reinicia o ambiente
# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configura URI do registro de modelos do MLflow para o Unity Catalog do Databricks
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,Configurações do modelo de propensão à inadimplência
model_name = "sicredi_coop_0903.int.modelo_detec_fraude"
suffix = "production"

# COMMAND ----------

import os
import mlflow.pyfunc

#configura o URI do modelo registradow, usa o (model_name) e a tag (suffix), result final: "models:/meu_modelo@Production"
model_uri = f"models:/{model_name}@{suffix}"

#retorna caminho do modelo
model_path = mlflow.pyfunc.get_model_dependencies(model_uri)
local_path = os.path.dirname(model_path)

#configura o caminho completo para o requeriments
requirements_path = os.path.join(local_path, "requirements.txt")
if not os.path.exists(requirements_path):
  dbutils.fs.put("file:" + requirements_path, "", True)

# COMMAND ----------

#instala os requeriments acima configurados
%pip install -r $requirements_path

#reinicia o ambiente python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,saveAsInt
def saveAsInt(dataframe, tipoCarga, p_data=None):
    if tipoCarga == 'recreate':
        sql(f"DROP TABLE IF EXISTS {table_name}")
        sql(f"CREATE TABLE IF NOT EXISTS {table_name}")
        dataframe.write.format("delta").option("overwriteSchema","true").mode("overwrite").saveAsTable(table_name)
    elif tipoCarga == 'delta':
        query = f"DELETE FROM {table_name} WHERE data_geracao = '{p_data}'"
        spark.sql(query)
        dataframe.write.format("delta").mode("append").saveAsTable(table_name)
    return dataframe.count()

# COMMAND ----------

# DBTITLE 1,process_table
def process_table(df,p_tipo_carga,p_data):
    result = saveAsInt(df, p_tipo_carga, p_data)  
    return result

# COMMAND ----------

# DBTITLE 1,Seleciona as linhas com antecedencia de vencimento = 15
# MAGIC %sql
# MAGIC select distinct * 
# MAGIC FROM sicredi_coop_0903.int.cartao_deploy

# COMMAND ----------

# DBTITLE 1,Guarda a seleção no df_deploy
df_deploy=_sqldf

# COMMAND ----------

# DBTITLE 1,Seleciona o modelo e o sufixo
model_name = "sicredi_coop_0903.int.modelo_detec_fraude"
suffix = "production"

# COMMAND ----------

# DBTITLE 1,Carrega e configura modelo com MLflow
import mlflow
from pyspark.sql.functions import struct
mlflow.set_registry_uri('databricks-uc')

model_uri = f"models:/{model_name}@{suffix}"

# create spark user-defined function for model prediction
#modelo_inatividade = mlflow.sklearn.load_model(model_uri)

modelo_fraude=mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# DBTITLE 1,Seleciona colunas específicas do dataframe df_deploy
#carrega o df_final com as informacoes do df_deploy
df_final = df_deploy.select("*")

#seleciona algumas colunas para retornar ao df posteriormente
CodAssociado = df_deploy.select('cpf_cnpj','id_cartao','ano_mes_ref','data_util_vencimento_fatura')

#lista de colunas selecionadas
colunas_selecionadas=["m2_pagamento_total", "m1_pagamento_total", "vlr_fatura", "principalidade", "risco",  "m3_pagamento_total", "perc_atual_lim_utilizado", "flg_inadimplente","debito_em_conta","isa","idade","media_perc_limite_utilizado","num_pagamentos_minimos", "vlr_saldo_variacao_3_M", "total_vlr_saldo", "atraso_pagamentos","razao_fatura_limite","evol_perc_limite_utilizado_m3_m1", "dias_sem_relacionamento"]

df_deploy = df_final.select(colunas_selecionadas).select("*")

# COMMAND ----------

# DBTITLE 1,Indexando e transformando coluna de inadimplência
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# Convertendo o DataFrame do Spark para Pandas
df_deploy_pandas = df_deploy.toPandas()

# Checando se df_deploy_pandas está vazio
# Supondo que 'modelo_inatividade' seja um modelo treinado em PySpark
# Prevendo as probabilidades
y_deploy_proba = modelo_fraude.predict_proba(df_deploy_pandas)[:,1]
y_deploy = (y_deploy_proba >= 0.90).astype('uint8')

# COMMAND ----------

# DBTITLE 1,Converte CodAssociado para Pandas
if df_deploy.count() > 0:
    CodAssociado = CodAssociado.toPandas()

# COMMAND ----------

# DBTITLE 1,Join CodAssociado em df_deploy_pandas

df_result = df_deploy_pandas

# COMMAND ----------

# DBTITLE 1,Atualiza df_result com dados de probabilidade
from datetime import datetime
if df_deploy.count() > 0:
    #df_result["flg_inadimplente"]=y_deploy
    #df_result["real"]=y_real
    df_result["probabilidade"]=y_deploy_proba
    df_result["data_geracao"]=datetime.now()

# COMMAND ----------

# DBTITLE 1,Cria spark df vindo do pandas df
# Assuming df_result is your existing Pandas DataFrame
if df_deploy.count() > 0:
    sdf_result = spark.createDataFrame(df_result)

# COMMAND ----------

# DBTITLE 1,Persistência e processamento da tabela de leads
#Persiste a Tabela
#Alterar para delta de data_geracao
if df_deploy.count() > 0:
    table_name = 'sicredi_coop_0903.exp.cartao_deploy_diaria'
    # Checando se df_deploy_pandas está vazio

    if not sdf_result.rdd.isEmpty():
        table_exists = spark.catalog.tableExists(table_name)
        if table_exists:
            process_table(sdf_result, 'delta', datetime.now().strftime('%Y-%m-%d'))
        else:
            process_table(sdf_result, 'recreate', datetime.now().strftime('%Y-%m-%d'))

# COMMAND ----------

if df_deploy.count() > 0:
    # Filtrar registros com probabilidade maior que 0.90
    df_filtered_high = sdf_result.filter(sdf_result.probabilidade > 0.90)
    
    # Filtrar registros com probabilidade menor ou igual a 0.90
    df_filtered_low = sdf_result.filter(sdf_result.probabilidade <= 0.90)
    
    # Agrupar por probabilidade e contar os registros
    df_grouped_high = df_filtered_high.groupBy("probabilidade").count().withColumnRenamed("count", "count_high")
    df_grouped_low = df_filtered_low.groupBy("probabilidade").count().withColumnRenamed("count", "count_low")
    
    # Unir os DataFrames
    df_grouped = df_grouped_high.union(df_grouped_low)
    
    # Converter o DataFrame do Spark para um DataFrame do Pandas
    df_pandas = df_grouped.toPandas()
    import matplotlib.pyplot as plt

    # Plotar os dados
    plt.figure(figsize=(8, 6))
    plt.bar(df_pandas['probabilidade'], df_pandas['count_high'], color='green', label='Probabilidade > 0.90')
    plt.bar(df_pandas['probabilidade'], df_pandas['count_low'], color='red', label='Probabilidade <= 0.90')
    plt.xlabel('Probabilidade')
    plt.ylabel('Quantidade de Registros')
    plt.title('Quantidade de Registros por Probabilidade')
    plt.legend()
    plt.show()