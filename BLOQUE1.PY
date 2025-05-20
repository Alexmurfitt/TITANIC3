# ----------------------------------------
# 🚢 TITANIC SOTA PIPELINE (2025) - BLOQUE 1
# Introducción, Setup y EDA SOTA
# ----------------------------------------

# 1. Configuración inicial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# 2. Cargar datos
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# 3. Primer vistazo rápido
display(train.head())
display(test.head())

# 4. Info general
print("=== TRAIN INFO ===")
train.info()
print("\n=== TEST INFO ===")
test.info()

print("\n=== Null values (train) ===")
print(train.isnull().sum())
print("\n=== Null values (test) ===")
print(test.isnull().sum())

# 5. Estadísticas descriptivas rápidas
display(train.describe(include='all'))
display(test.describe(include='all'))

# 6. Visualización de missing values
msno.matrix(train)
plt.title('Missing values in train set')
plt.show()

msno.bar(train)
plt.title('Missing values per feature (train)')
plt.show()

msno.matrix(test)
plt.title('Missing values in test set')
plt.show()

# 7. Distribución de la variable objetivo
plt.figure(figsize=(6,3))
sns.countplot(x='Survived', data=train)
plt.title('Distribución de la variable objetivo (Survived)')
plt.show()

# 8. Visualización rápida de correlaciones numéricas
plt.figure(figsize=(12,7))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlación entre variables numéricas')
plt.show()
