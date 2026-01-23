import pandas as pd
from load_data import load_data

df = load_data("data.csv")

print("Sample Data")
print(df.head())

print("\nCount of Null Values")
print(df.isnull().sum())

print("\nInfo of the data")
print(df.info())

print("\nData Description")
print(df.describe())