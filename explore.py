# Instala si no lo tienes: pip install ydata-profiling
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('train_clean.csv')
profile = ProfileReport(df, title="Reporte Avanzado Titanic", explorative=True)
profile.to_file('Titanic_EDA_Advanced_Report.html')
