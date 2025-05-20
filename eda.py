# 1. CARGA Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Dimensiones Train:", train.shape)
print("Dimensiones Test :", test.shape)

# Revisar primeras filas
print(train.head())
print(test.head())

# Tipos y nulos
print("\n--- Tipos de datos y nulos ---")
print(train.info())
print("\nNulos por columna:")
print(train.isnull().sum())

# Estadísticas descriptivas generales
print("\n--- Estadísticas Descriptivas ---")
print(train.describe(include='all'))

# Visualización rápida de nulos
sns.heatmap(train.isnull(), cbar=False, cmap="YlGnBu")
plt.title("Mapa de nulos (train.csv)")
plt.show()

# Distribución de la variable objetivo
plt.figure(figsize=(4,3))
train['Survived'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Distribución Variable Objetivo: Survived')
plt.xticks([0,1], ['No sobrevivió', 'Sobrevivió'], rotation=0)
plt.ylabel('Frecuencia')
plt.show()

# Variables categóricas: conteo de valores únicos
for col in train.columns:
    if train[col].dtype == 'object':
        print(f"\nColumna '{col}': {train[col].nunique()} categorías")
        print(train[col].value_counts())

# Distribución de edades y sexo vs supervivencia
plt.figure(figsize=(8,4))
sns.histplot(data=train, x='Age', bins=30, kde=True, hue='Survived', element='step')
plt.title('Distribución de Edad por Supervivencia')
plt.show()

sns.countplot(data=train, x='Sex', hue='Survived')
plt.title('Supervivencia por Sexo')
plt.show()

# Revisión rápida de correlaciones
plt.figure(figsize=(8,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, fmt=".2f", cmap='RdBu')
plt.title('Correlación entre variables numéricas')
plt.show()
