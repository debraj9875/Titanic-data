
# 1. Import libraries and dataset, explore basic info
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')


# Explore dataset info and null counts
print(df.info())
print(df.isnull().sum())

# 2. Handle missing values
# For 'age' (numerical), fill missing values with median
df['age'].fillna(df['age'].median(), inplace=True)

# For 'embarked' (categorical), fill missing with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop columns with too many missing values or irrelevant for ML
df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], inplace=True)

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# 3. Convert categorical features into numerical using encoding
# For categorical variables: 'sex', 'embarked'
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# 4. Normalize/standardize numerical features
scaler = StandardScaler()
num_cols = ['age', 'fare', 'sibsp', 'parch']

df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Visualize outliers using boxplots and remove them
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Define a function to remove outliers based on IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Remove outliers iteratively for all numerical columns
for col in num_cols:
    df = remove_outliers(df, col)

# Final dataset info
print(df.info())
print(df.head())
