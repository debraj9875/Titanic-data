import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)

# 2. Preview data & check info
print(df.head())
print(df.info())
print(df.isnull().sum())  # Check missing value counts

# 3. Fill missing Age values with the median, Fare with the mean (if present)
if df['Age'].isnull().sum() > 0:
    df['Age'] = df['Age'].fillna(df['Age'].median())
if df['Fare'].isnull().sum() > 0:
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# 4. Encode categorical 'Sex' column
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 5. Normalize/standardize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 6. Visualize outliers using boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.show()

# 7. Remove outliers in 'Fare' using IQR
Q1_fare = df['Fare'].quantile(0.25)
Q3_fare = df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
df = df[(df['Fare'] >= Q1_fare - 1.5 * IQR_fare) & (df['Fare'] <= Q3_fare + 1.5 * IQR_fare)]

# Remove outliers in 'Age' using IQR
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
df = df[(df['Age'] >= Q1_age - 1.5 * IQR_age) & (df['Age'] <= Q3_age + 1.5 * IQR_age)]

# Final overview
print(df.info())
print(df.head())
