import pandas as pd
df = pd.read_csv("submission.csv")
print(df.isnull().sum())
print(df["Survived"].value_counts())
