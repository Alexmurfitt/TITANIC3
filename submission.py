import pandas as pd

submission = pd.read_csv('submission.csv')
print("Primeras filas del submission:")
print(submission.head())

print("\nDistribución de predicciones:")
print(submission['Survived'].value_counts())
