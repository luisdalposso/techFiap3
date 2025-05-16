# Databricks notebook source
# Instalar pacotes necessários
# Execute no terminal ou como célula de código no notebook
!pip install pandas matplotlib imbalanced-learn pyspark

# Imports necessários para o código
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Restante do código já atualizado

# COMMAND ----------

# Configurar credenciais do Kaggle
dbutils.fs.put("/FileStore/kaggle.json", """
{
  "username":"ldalposso",
  "key":"312105743e1a687ce7f7df7a52fcee4a"
}
""", overwrite=True)

dbutils.fs.mkdirs("file:/root/.kaggle")
dbutils.fs.cp("/FileStore/kaggle.json", "file:/root/.kaggle/kaggle.json")

# COMMAND ----------

# MAGIC %sh
# MAGIC chmod 600 /root/.kaggle/kaggle.json
# MAGIC ls -la /root/.kaggle/  # Verificação

# COMMAND ----------

#testar api
!pip install kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate() 
print("Autenticação bem-sucedida!")

# COMMAND ----------

from kaggle.api.kaggle_api_extended import KaggleApi
import os

# 1. Configurações
DATASET = "mlg-ulb/creditcardfraud" 
LOCAL_DIR = "/dbfs/FileStore/creditcard"
LOCAL_FILE = f"{LOCAL_DIR}/creditcard.csv"

# 2. Criar diretório e baixar dados
try:
    print("1. Preparando ambiente...")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print("2. Autenticando na API Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    print("3. Iniciando download...")
    api.dataset_download_files(DATASET, path=LOCAL_DIR, unzip=True, force=True)  # Corrigido aqui
    
    # Verificação
    if not os.path.exists(LOCAL_FILE):
        raise FileNotFoundError(f"Arquivo não encontrado em {LOCAL_FILE}")
    
    print(f"4. Download concluído! Tamanho: {os.path.getsize(LOCAL_FILE)/1e6:.2f} MB")

except Exception as e:
    print(f"\n❌ ERRO: {str(e)}")
    print("Soluções possíveis:")
    print("- Verifique se aceitou os termos do dataset em kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("- Confira o arquivo /root/.kaggle/kaggle.json")
    raise

# 3. Carregar para DataFrame (usando caminho local confirmado)
try:
    print("\n5. Carregando DataFrame...")
    df = spark.read.csv(f"file:{LOCAL_FILE}", header=True, inferSchema=True)
    
    print(f"6. Sucesso! Total de linhas: {df.count()}")
    display(df.limit(3))

except Exception as e:
    print(f"\n❌ ERRO AO CARREGAR: {str(e)}")
    print("Execute manualmente para diagnóstico:")
    print(f"%sh ls -la {LOCAL_DIR}")
    raise

# 4. Persistência (opcional)
try:
    df.write.format("delta").mode("overwrite").saveAsTable("credit_card_fraud")
    print("\n7. Tabela 'credit_card_fraud' criada com sucesso!")
except Exception as e:
    print(f"\n⚠️ AVISO: Não foi possível persistir ({str(e)})")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM credit_card_fraud LIMIT 5