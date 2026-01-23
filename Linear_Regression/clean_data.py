import pandas as pd
from load_data import load_data

df = load_data("data.csv")

# dropping rows where target value is missing
df = df.dropna(subset=['salary'])
print(df['salary'].isnull())

# handling outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# handling/filling numerical values
df['age'] = df['age'].fillna(df['age'].median())
df['experience'] = df['experience'].fillna(df['experience'].median())
df['working_hours'] = df['working_hours'].fillna(df['working_hours'].median())
df['certifications'] = df['certifications'].fillna(df['certifications'].median())

# Fill categorical columns
df['education_level'] = df['education_level'].fillna(df['education_level'].mode()[0])
df['city'] = df['city'].fillna(df['city'].mode()[0])

df = remove_outliers_iqr(df, "salary")

print(df.max)

df.to_csv("cleaned_data.csv", index=False)