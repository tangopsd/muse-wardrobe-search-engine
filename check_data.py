import pandas as pd

df = pd.read_csv("data/styles.csv", on_bad_lines="skip")
print(df.shape)
print(df.columns.tolist())
print(df[["masterCategory", "subCategory", "articleType"]].value_counts().head(20))