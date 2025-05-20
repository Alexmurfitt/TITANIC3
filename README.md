# README 1
# 🚢 Titanic - Machine Learning from Disaster

## 📌 **Objetivo del Proyecto**

Este proyecto aborda el problema clásico de predicción de supervivencia en el desastre del Titanic utilizando técnicas avanzadas de Machine Learning, validación rigurosa y pipeline reproducible. El propósito es lograr la **mayor precisión posible** en la predicción de la variable objetivo (`Survived`), implementando un flujo de trabajo profesional, sólido y alineado con las mejores prácticas de ciencia de datos y competición en plataformas como Kaggle.

El objetivo principal es:

* **Construir un sistema de predicción de supervivencia** que generalice de manera óptima a datos no vistos, evitando overfitting, fugas de información (leakage) y errores comunes.
* **Utilizar métodos de evaluación y selección de modelos de última generación**, como Stratified K-Fold Cross Validation, ensamblados y optimización de hiperparámetros con Optuna.
* **Documentar, automatizar y explicar** todos los pasos para reproducibilidad, robustez y transparencia, permitiendo futuras extensiones y experimentos.

---

## 🧭 **Estrategia General y Plan de Trabajo**

La estrategia del proyecto está dividida en **fases claramente delimitadas**:

### 1. **Carga y Análisis Exploratorio de Datos (EDA)**

* Carga de los datasets `train.csv` y `test.csv`.
* Análisis de dimensiones, primeras filas y tipos de variables.
* Revisión exhaustiva de valores nulos por columna y visualización de patrones de missing data (heatmaps).
* Estadísticas descriptivas de variables numéricas y categóricas.
* Distribución de la variable objetivo (`Survived`).
* Análisis gráfico de variables clave como edad y sexo respecto a la supervivencia.
* Matriz de correlación entre variables numéricas.
* **Estatus:** ***COMPLETADO***

### 2. **Preprocesamiento y Feature Engineering**

* **Imputación de valores nulos:**

  * Edad (`Age`): imputar por media/mediana o métodos más sofisticados (regresión, KNN).
  * Cabina (`Cabin`): alta proporción de nulos, suele eliminarse o transformarse en una feature binaria (“Cabin\_known”).
  * Embarque (`Embarked`): imputar por la moda.
* **Ingeniería de variables:**

  * Extracción de títulos del campo `Name` (Mr, Mrs, Miss...).
  * Agrupación y binarización de variables categóricas (`Sex`, `Embarked`).
  * Combinación de `SibSp` y `Parch` en una sola feature de “familia” o “is\_alone”.
  * Tratamiento de tickets y cabinas para identificar patrones útiles.
* **Codificación y escalado:**

  * Codificación LabelEncoder o OneHot para variables categóricas.
  * Escalado robusto de variables numéricas para modelos sensibles.
* **Estatus:** ***EN PROCESO (siguiente paso inmediato)***

### 3. **División de datos y Estrategia de Evaluación**

* **Separación de features y variable objetivo:**
  Nunca utilizar datos de `test.csv` para ninguna fase de entrenamiento o validación.
* **Validación Cruzada Estratificada (StratifiedKFold):**

  * Estratificación para preservar la proporción de clases en cada fold.
  * Uso de 5 o 10 folds para reducir la varianza y evitar overfitting.
  * Toda la selección de modelos e hiperparámetros se hará solo con `train.csv`.
* **Estatus:** ***Planificado para la fase de modelado***

### 4. **Modelado Avanzado**

* **Modelos base:**

  * LightGBM, XGBoost, CatBoost, RandomForest, y opcionalmente TabNet y AutoML (AutoGluon).
* **Optimización de hiperparámetros:**

  * Búsqueda bayesiana con Optuna en ciclo de cross-validation.
  * Early stopping y análisis de overfitting.
* **Ensamblado de modelos:**

  * StackingClassifier con los mejores modelos base.
  * Meta-learner robusto (RandomForest, LogisticRegression).
* **Evaluación interna:**

  * Media y desviación estándar de accuracy/F1/ROC-AUC por fold.
  * Registro de hiperparámetros y resultados para trazabilidad.
* **Estatus:** ***Pendiente, programado tras preprocesamiento***

### 5. **Interpretabilidad y Análisis de Importancia**

* **Análisis SHAP:**

  * Identificación de las features más importantes para la predicción de supervivencia.
  * Visualizaciones para interpretabilidad.
* **Estatus:** ***Pendiente***

### 6. **Predicción final y generación de submission**

* **Entrenamiento final:**

  * Reentrenar el modelo óptimo con TODO `train.csv` y mejores hiperparámetros.
* **Predicción sobre test.csv:**

  * Uso exclusivo de test.csv para predicción, nunca para tuning ni visualización previa.
* **Generación de archivo submission.csv** con formato requerido por Kaggle.
* **Estatus:** ***Pendiente, último paso***

---

## ✅ **¿Qué se ha hecho ya?**

* Instalación y configuración avanzada de entorno virtual y dependencias (scikit-learn, lightgbm, catboost, xgboost, optuna, shap, seaborn, matplotlib, pandas, numpy, pytorch-tabnet, autogluon).
* EDA completo con:

  * Revisión de nulos.
  * Estadísticas descriptivas.
  * Visualizaciones gráficas (heatmaps, histogramas, countplots, correlaciones).
  * Análisis de variables categóricas y numéricas.

---

## 🛠️ **¿Qué falta por hacer? (Plan detallado)**

1. **Imputar nulos de forma robusta y documentar decisiones de tratamiento.**
2. **Construir features avanzadas, inspiradas en la literatura de competiciones y análisis de supervivencia.**
3. **Implementar pipelines reproducibles de preprocesamiento y modelado usando sklearn.pipeline o frameworks similares.**
4. **Configurar y ejecutar Validación Cruzada Estratificada para cualquier experimento de modelado.**
5. **Optimizar modelos con Optuna, y ensamblar varios modelos top (stacking).**
6. **Realizar interpretabilidad avanzada con SHAP y otras técnicas.**
7. **Entrenar el modelo final y generar las predicciones en test.csv, creando el submission.**
8. **Registrar resultados y generar documentación de cada experimento para reproducibilidad total (opcional: MLflow).**

---

## 🗺️ **Roadmap Visual**


graph TD
    A[EDA & Diagnóstico] --> B[Preprocesamiento y Feature Engineering]
    B --> C[Evaluación y Validación Cruzada]
    C --> D[Modelado y Optimización]
    D --> E[Interpretabilidad y Ensamblado]
    E --> F[Entrenamiento final y Predicción en test.csv]
    F --> G[Generación de Submission y Documentación]


---

## 🚀 **Notas y mejores prácticas**

* **No uses datos de test.csv para nada antes del final.**
* **Controla la reproducibilidad:** fija random\_state y documenta todos los experimentos.
* **Evalúa siempre en validación interna (no en test) hasta el último paso.**
* **Registra resultados, hiperparámetros y decisiones para poder iterar y mejorar.**
* **Incluye análisis interpretativo para entender y explicar el modelo final.**

---

## 🔗 **Referencias y Recursos**

* [Guía oficial Titanic Kaggle](https://www.kaggle.com/c/titanic)
* [Stacked Generalization (Wolpert, 1992)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=45ee02a5c440d8d282e088b4dba4a27e8581e1dc)
* [Optuna: A hyperparameter optimization framework](https://optuna.org/)
* [SHAP values for model interpretability](https://shap.readthedocs.io/en/latest/)
* [Feature Engineering Strategies for Titanic](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial)

# 🆕 README 2 — ESTADO ACTUALIZADO
# 🚢 Titanic - Machine Learning from Disaster (Versión SOTA 2025)
📌 Objetivo del Proyecto
El objetivo es construir el pipeline más avanzado, robusto y reproducible para la predicción de supervivencia en el Titanic. El workflow sigue los estándares más altos de ciencia de datos:

EDA profundo

Feature engineering manual y automático

Imputación avanzada

Modelado con ensamblado y optimización hiperparamétrica

Interpretabilidad SHAP

Documentación paso a paso

Preparado para competiciones tipo Kaggle

⚙️ Estrategia SOTA: Resumen de Bloques
Introducción y configuración

EDA SOTA (Exploratory Data Analysis)

Feature Engineering manual y automático (Deep Feature Synthesis)

Imputación avanzada de missing values

Codificación y escalado

Selección de variables avanzada

Modelado avanzado + Optimización hiperparámetros (Optuna)

Stacking/blending ultra-avanzado

Interpretabilidad (SHAP global y local)

AutoML y blending externo

Exportación reproducible y README

Apéndice: recomendaciones, troubleshooting y mejoras

# 1. Introducción, Setup y EDA SOTA ✔️ (EJECUTADO). Carga y Análisis Exploratorio de Datos (EDA) ✔️ (EJECUTADO)

- Carga de datos:train.csv y test.csv.
- Análisis general y visualización de missing values: Análisis de dimensiones, primeras filas y tipos de variables.
- Estadísticas descriptivas
- Distribución de la variable objetivo
- Correlaciones visuales
- Revisión exhaustiva de valores nulos (missingno, pandas).
- Estadísticas descriptivas numéricas y categóricas.
- Visualización de la variable objetivo.
- Matriz de correlación de variables numéricas.


# 2. Feature Engineering manual y Deep Feature Synthesis 🟡 (SIGUIENTE PASO)
Extracción de variables avanzadas (Título, Familia, TicketGroup, Deck, Binning)

Generación automática de features sintéticas (usando featuretools)

Análisis de importancia preliminar

# 3. Imputación avanzada de missing values 🔲
Métodos SOTA: KNN, regresión, MICE, valores categóricos especiales, etc.

# 4. Codificación y escalado 🔲
Encoding categórico robusto (Label, OneHot, Target, Ordinal, etc.)

Escalado robusto y selección de técnicas según modelo

# 5. Selección de variables avanzada 🔲
Permutation importance, SHAP, correlaciones, leakage check

# 6. Modelado avanzado + Optimización (Optuna) 🔲
Modelos: LightGBM, CatBoost, XGBoost, TabNet (opcional)

Optimización hiperparámetros, validación cruzada, reproducibilidad

# 7. Stacking/blending ultra-avanzado 🔲
Ensamblado con sklearn, mlxtend, blending externo y meta-learner

# 8. Interpretabilidad SHAP 🔲
Análisis global y local

Gráficas, importancia y explicación detallada

# 9. AutoML y blending externo 🔲
Benchmark con AutoGluon, H2O.ai, etc.

# 10. Exportación reproducible y README 🔲
Generación de submission.csv, guardado de modelos y artefactos reproducibles

README.md actualizado con resultados y decisiones

# 11. Apéndice
Recomendaciones, troubleshooting, ideas para mejoras

🛠️ Histórico y Estado de Ejecución
 EDA profesional (Bloque 1) ejecutado

 Feature Engineering avanzado (Bloque 2)

 Imputación avanzada (Bloque 3)

 Codificación y escalado (Bloque 4)

 Selección de variables (Bloque 5)

 Modelado y optimización (Bloque 6)

 Stacking/blending (Bloque 7)

 Interpretabilidad SHAP (Bloque 8)

 AutoML (Bloque 9)

 Exportación y README final (Bloque 10)

 Apéndice (Bloque 11)

📦 Requisitos

pip install pandas numpy matplotlib seaborn missingno featuretools category_encoders scikit-learn lightgbm catboost xgboost optuna shap mlxtend autogluon.tabular pytorch-tabnet
📂 Estructura Recomendada de Carpetas

titanic/
│
├── TITANIC_SOTA_PIPELINE_2025.ipynb
├── train.csv
├── test.csv
├── gender_submission.csv
├── submission.csv
├── BLOQUE1.py / eda_sota.py (si trabajas en .py)
├── feature_engineering.py
├── ...
│
├── models/
├── plots/
├── README.md
└── requirements.txt

# README 3 
# 🚢 Titanic - Machine Learning from Disaster (SOTA 2025)

## 📌 Objetivo del Proyecto

* **Predecir la supervivencia** de pasajeros usando un sistema robusto, modular y explicable.
* **Maximizar precisión y generalización** en datos no vistos, evitando fugas y sobreajuste.
* **Pipeline reproducible, automatizable y documentado** para futuras mejoras, investigación y competición.

---

## 🧭 Estrategia General

Dividido en bloques y fases:

1. **EDA SOTA**: Exploración avanzada, visual, y diagnóstico de datos.
2. **Feature Engineering**: Manual + automática (Deep Feature Synthesis).
3. **Imputación avanzada**: Métodos robustos para nulos.
4. **Codificación y Escalado**: Según modelo y sensibilidad.
5. **Selección de Features**: Basada en importancia y leakage.
6. **Modelado y Ensembles**: LightGBM, CatBoost, XGBoost, stacking, blending y AutoML.
7. **Optimización de Hiperparámetros**: Optuna, búsqueda bayesiana.
8. **Interpretabilidad (Explainable AI)**: SHAP, análisis global y local.
9. **Exportación reproducible**: Submission, modelos y scripts.

---

## 🛠️ Progreso y Tareas

| Bloque                                       | Estado       | Comentario breve                                |
| -------------------------------------------- | ------------ | ----------------------------------------------- |
| 1. EDA SOTA                                  | EN PROCESO   | Notebook en ejecución, falta instalar missingno |
| 2. Feature Engineering manual y automática   | PENDIENTE    |                                                 |
| 3. Imputación avanzada de nulos              | PENDIENTE    |                                                 |
| 4. Codificación y escalado                   | PENDIENTE    |                                                 |
| 5. Selección avanzada de variables           | PENDIENTE    |                                                 |
| 6. Modelado, hiperparametrización, ensembles | PENDIENTE    |                                                 |
| 7. Interpretabilidad SHAP                    | PENDIENTE    |                                                 |
| 8. Exportación, reproducibilidad             | PENDIENTE    |                                                 |
| 9. README y documentación                    | ACTUALIZANDO | Se va actualizando por bloque                   |

---

## ✅ ¿Qué se ha ejecutado?

* Configuración entorno virtual (venv\_titanic)
* Instalación dependencias ML: pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm, xgboost, catboost, optuna, shap
* Carga y revisión inicial de los archivos (train.csv, test.csv, submission.csv)
* Script de inspección rápida de submission
* EDA SOTA: pendiente solo instalar y ejecutar `missingno` para gráficos de nulos

---

## 🔜 Próximos pasos

1. Instalar **missingno** para terminar EDA visual.
2. Ejecutar **Bloque 2: Feature Engineering manual y automático** (featuretools).
3. Documentar decisiones y resultados en este README.
4. Continuar secuencialmente con los siguientes bloques, garantizando reproducibilidad y máxima precisión en cada fase.

# README 4:
# 🚢 Titanic SOTA Pipeline (2025)

📌 Objetivo

Desarrollar el sistema más avanzado, robusto y reproducible para predecir la supervivencia en el Titanic, integrando EDA SOTA, Feature Engineering manual y automático, imputación avanzada, modelado ensemble, optimización, interpretabilidad y exportación reproducible.

🧭 Fases y Progreso

1. EDA SOTA (Exploratory Data Analysis) ✅

Carga de train.csv y test.csv.

Análisis de dimensiones, info, valores nulos y primeras filas (comprobado: train (891, 12), test (418, 11)).

Estadísticas descriptivas y visualización de nulos (missingno).

Gráficos de distribución de la variable objetivo y correlaciones numéricas.

Estado: Completado (outputs confirmados por usuario).

2. Feature Engineering Manual y Automático ⏳

Extracción de título (Title), tamaño familiar, grupos de ticket, deck/cabina, binning de edad/fare, etc.

Preparar Deep Feature Synthesis con featuretools para generación de features automáticas.

Visualización y análisis de nuevas features.

Estado: En progreso (siguiente bloque tras EDA).

3. Imputación Avanzada de Missing Values ⏳

Imputar edad por mediana/grupo o regresión; cabina como binario/Deck; embarque por moda.

Documentar cada imputación y justificar elección.

Estado: A realizar tras feature engineering.

4. Codificación y Escalado ⏳

Label Encoding y OneHot para categóricas según el modelo.

Escalado robusto de numéricas (opcional para árboles, necesario para DL).

Estado: A continuación.

5. Selección de Variables Avanzada ⏳

Importancia de features (Random Forest, SHAP, Permutation Importance).

Eliminación de leakage y variables irrelevantes.

Estado: Tras Feature Engineering.

6. Modelado Avanzado + Optimización ⏳

Modelos: LightGBM, CatBoost, XGBoost, RandomForest, opcional TabNet/AutoML.

Tuning hiperparámetros con Optuna.

Cross-validation estratificada (StratifiedKFold).

Estado: Tras features y selección.

7. Stacking/Blending Ultra-avanzado ⏳

Ensemble con 3+ modelos base y meta-learner robusto.

Blending y voting, benchmark con AutoML (AutoGluon).

Estado: Tras modelado base.

8. Interpretabilidad (SHAP global y local) ⏳

Visualización de importancia global/local (summary plots, force plots).

Análisis de errores y casos clave.

Estado: Tras modelo final/ensemble.

9. Exportación Reproducible y README ⏳

Submission Kaggle en formato requerido.

Guardado de modelos, scalers y seeds fijos.

Actualización continua de este README y scripts para máxima trazabilidad.

10. Apéndice: recomendaciones y troubleshooting ⏳

Registro de ideas, mejoras, alternativas y hallazgos.

🚩 Últimos outputs revisados

Train shape: (891, 12) / Test shape: (418, 11)

Primeras filas mostradas (Name, Age, Fare, Embarked, etc.).

Valores nulos confirmados en Cabin y Age/Embark
familia, deck, ticketgroup, bins, etc.).

Bloque 2b: Feature Engineering automático con featuretools (Deep Feature Synthesis).

Documentar e ilustrar todo en este README.

Ejecutar bloque de imputación avanzada y documentar estrategias.

Continuar con pipeline bloque a bloque hasta stacking, interpretabilidad y exportación.

¿Cómo continuar?

Ejecutar los bloques en orden, revisando outputs y anotando todo avance aquí.

Actualizar README tras cada fase (outputs clave, decisiones y próximos pasos).

En caso de error o nueva hipótesis, registrar el troubleshooting y solución aplicada.

✍️ Log de ejecución

BLOQUE 1 ejecutado y outputs confirmados visualmente.

Listo para Feature Engineering y documentación de todas las nuevas variables generadas.

Esperando ejecución de Bloque 2 (Manual + Automático).

# README 
🚢 Titanic - Machine Learning from Disaster (Pipeline SOTA 2025)

📌 Objetivo

Construir el sistema más avanzado y reproducible de predicción de supervivencia en el Titanic usando técnicas SOTA de machine learning y ciencia de datos tabulares, alineado con las mejores prácticas internacionales y los estándares de competición Kaggle.

📚 Bloques del pipeline

Introducción y configuración

Análisis Exploratorio de Datos (EDA) SOTA

Feature Engineering manual y automático (Deep Feature Synthesis)

Imputación avanzada de missing values

Codificación y escalado

Selección de variables avanzada

Modelado avanzado + Optimización de hiperparámetros (Optuna)

Stacking/blending ultra-avanzado

Interpretabilidad (SHAP global y local)

AutoML y blending externo

Exportación reproducible y README

Apéndice: troubleshooting y recomendaciones

Estado del pipeline y checklist

Bloque

Estado

Introducción y setup

✅ COMPLETO

EDA SOTA (análisis y visualización)

✅ COMPLETO

Feature engineering manual

⬜ PENDIENTE

Deep Feature Synthesis (featuretools)

⬜ PENDIENTE

Imputación missing values avanzada

⬜ PENDIENTE

Codificación y escalado

⬜ PENDIENTE

Selección de variables avanzada

⬜ PENDIENTE

Modelado: LightGBM, CatBoost, XGBoost

⬜ PENDIENTE

Optuna + validación cruzada robusta

⬜ PENDIENTE

Stacking/blending avanzado

⬜ PENDIENTE

Interpretabilidad SHAP

⬜ PENDIENTE

AutoML y blending externo

⬜ PENDIENTE

Exportación y documentación

⬜ PENDIENTE

Última actualización: (rellenar fecha después de terminar cada bloque)

# ### Variables de Realismo Histórico

Se ha incorporado explícitamente una feature de "prioridad de salvamento histórica" que modela las reglas y criterios oficiales de evacuación del Titanic:

- Mujeres y niños primero, sin importar la clase.
- Prioridad intermedia para hombres de primera clase.
- Prioridad más baja para hombres adultos de segunda y tercera clase.

Esta feature permite al modelo aprender y reproducir los patrones sociales, históricos y operativos documentados oficialmente durante el desastre, asegurando un sistema predictivo alineado al máximo con la realidad y explicable en profundidad.

### 🛟 Feature de Prioridad de Salvamento Histórica

Se ha añadido una variable sintética `RescuePriority`, diseñada en base a los criterios oficiales aplicados durante el desastre del Titanic:

- Valor 3: Mujeres y niños (<15 años) de cualquier clase.
- Valor 2: Hombres adultos de 1ª clase.
- Valor 1: Hombres adultos de 2ª y 3ª clase.

Esta variable permite al modelo aprender el patrón real de supervivencia, alineando el sistema predictivo con los hechos históricos verificados y los estudios oficiales del naufragio.

# Durante el hundimiento del Titanic en la madrugada del 15 de abril de 1912, los criterios principales que se aplicaron para determinar la prioridad de salvamento en los botes salvavidas se pueden resumir en los siguientes puntos fundamentales:

1. Prioridad: “Mujeres y niños primero”
Norma social predominante: La regla de “Mujeres y niños primero” era el criterio oficial y socialmente aceptado en la época para situaciones de naufragio.

Aplicación desigual: Aunque esta regla fue anunciada y promovida por la tripulación (especialmente los oficiales), su aplicación fue desigual en distintas partes del barco y según los oficiales a cargo de cada bote.

2. Clase del pasajero
División por clases: El Titanic tenía una marcada segregación por clases (Primera, Segunda y Tercera clase).

Acceso a los botes: Los pasajeros de Primera clase tuvieron mucho mayor acceso y posibilidades de ser evacuados que los de Segunda, y sobre todo que los de Tercera.

Obstáculos físicos: Los pasajeros de Tercera clase a menudo se encontraron con puertas cerradas o con mayor dificultad para llegar a las cubiertas donde estaban los botes.

Prioridad práctica: En la práctica, la clase social influyó fuertemente en las posibilidades de supervivencia.

3. Ubicación en el barco
Proximidad a los botes: Los pasajeros y tripulantes que se encontraban cerca de las cubiertas superiores (donde estaban los botes) tuvieron mayor oportunidad de embarcarse.

Retraso en la información: Muchos pasajeros de Tercera clase no recibieron la información o la alarma a tiempo, lo que disminuyó sus opciones de evacuación.

4. Rol de la tripulación
Tripulación esencial: Algunos miembros de la tripulación (especialmente marineros y oficiales) tenían prioridad para ocupar los botes como encargados de remarlos y dirigirlos, pero su número en los botes debía ser el mínimo necesario.

Tripulación de servicio: Otros tripulantes no esenciales no tenían prioridad y su supervivencia dependió, al igual que los pasajeros, de su acceso y situación.

5. Interpretación del “Mujeres y niños primero”
Variación por oficial: Algunos oficiales interpretaron la orden como “sólo mujeres y niños”, mientras otros permitieron a hombres subir si no había más mujeres o niños a la vista.

Ejemplo: El Oficial Murdoch permitió que algunos hombres subieran cuando no quedaban más mujeres o niños en su zona.

Ejemplo contrario: El Oficial Lightoller aplicó estrictamente la norma y prohibió que hombres adultos subieran, incluso si había espacio en los botes.

6. Otros factores sociales y de idioma
Idioma y nacionalidad: Algunos pasajeros de Tercera clase, extranjeros, no comprendieron las instrucciones dadas en inglés y perdieron tiempo o no supieron cómo actuar.

Desinformación y pánico: El miedo y el caos influyeron en la capacidad de algunos pasajeros para llegar a los botes.

Resumen esquemático
Prioridad oficial y práctica:

Mujeres y niños (todas las clases) — prioridad máxima.

Pasajeros de Primera clase (especialmente mujeres y niños).

Pasajeros de Segunda clase (especialmente mujeres y niños).

Pasajeros de Tercera clase (muchos quedaron atrapados o desinformados).

Hombres adultos (especialmente de Primera clase, solo si no había mujeres/niños cerca).

Miembros esenciales de la tripulación para manejar los botes.

En la práctica, la combinación de sexo, edad, clase social, ubicación, idioma y el oficial responsable de la evacuación determinó la supervivencia.

# Referencias
“Titanic: Voices from the Disaster” de Deborah Hopkinson.

“A Night to Remember” de Walter Lord.

Reportes oficiales de la Comisión Británica y de EE. UU. sobre el desastre del Titanic.