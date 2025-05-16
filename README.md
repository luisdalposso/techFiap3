## 📘 Notebook: 01_get_dados.py 01_get_dados.py 
Tem objetivo em fazer a coleta de dados utilizando a API do Kaggle, baixando o dataset de detecção de fraudes em cartão de crédito e carregá-lo em um Dataframe Spark e persisti-lo em uma tabela delta para consumo futuro.
---
<details>
  <summary><strong>📖 Clique para expandir a explicação do notebook</strong></summary>
  
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
