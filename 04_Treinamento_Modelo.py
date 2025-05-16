# Databricks notebook source
# MAGIC %md
# MAGIC ## Treinamento do Modelo

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             f1_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

df_treinamento = spark.read.table("credit_card_fraud_modelo")

# COMMAND ----------

df_treinamento.display()

# COMMAND ----------

# Converter DataFrame do PySpark para pandas
df_treinamento_pd = df_treinamento.toPandas()


# COMMAND ----------

# Features e target
X = df_treinamento_pd.drop(columns=['Class'])
y = df_treinamento_pd['Class']


# COMMAND ----------

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# COMMAND ----------

# Mostrar os DataFrames resultantes
print(X_train)
print(X_test)
print(y_train)
print(y_test)


# COMMAND ----------

# Definir o pipeline
pipeline = Pipeline([
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])



# COMMAND ----------

# Espaço de busca
param_dist = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0]
}

# COMMAND ----------

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# COMMAND ----------

# Random Search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    scoring='roc_auc',
    n_iter=10,
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# COMMAND ----------

# Treinar
search.fit(X_train, y_train)
best_model = search.best_estimator_

# COMMAND ----------

best_model

# COMMAND ----------

# Predições
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# COMMAND ----------

# -----------------------------
# AVALIAÇÕES
# -----------------------------

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# F1-Score
print("F1-Score:", f1_score(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC AUC:", roc_auc)

# Average Precision Score
avg_precision = average_precision_score(y_test, y_proba)
print("Average Precision Score (PR AUC):", avg_precision)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Feature importance
xgb_model = best_model.named_steps['clf']
importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importance[:10], y=importance[:10].index)
plt.title("Top 10 Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.show()


# COMMAND ----------

import os
os.listdir('/dbfs')

# COMMAND ----------

import pickle

filename = 'modelofraude.sav'
with open('/dbfs/FileStore/modelofraude.pkl', 'wb') as arquivo:
    pickle.dump(search, arquivo)