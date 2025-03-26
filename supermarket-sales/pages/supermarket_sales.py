import os 
filepath = os.path.join("phython project\supermarket-sales\supermarket_sales - Sheet1.csv")
import pandas as pd 
df=pd.read_csv(filepath)

# first we check  missing value

mising_value=df.isnull().sum()
# print(mising_value)

# Data Summary

numerical_summary = df.describe()
print(numerical_summary)

# 0utliers Check Using Boxplot

import seaborn as sns
import matplotlib.pyplot as plt 

# plt.figure(figsize=(8,5))
# sns.boxplot(x=df['Total'])
# plt.title("Boxplot of Total Sales (Outliers Check)")
# plt.show()

# Outlier detection ke liye numeric columns select karna
numeric_cols = ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income", "Rating"]

# IQR Method se outliers remove karna
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)  # 25% quantile
    Q3 = df[col].quantile(0.75)  # 75% quantile
    IQR = Q3 - Q1  # IQR calculate
    
    lower_bound = Q1 - 1.5 * IQR  # Lower bound
    upper_bound = Q3 + 1.5 * IQR  # Upper bound
    
    # Sirf wohi rows rakho jo is range ke andar hain
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Outliers remove hone ke baad naya dataset check karna

# print(df.describe())

# Sabhi numerical columns ka histogram plot karein
df.hist(figsize=(12, 8), bins=20)
plt.show()

#outlier checking 

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["Total", "Unit price", "gross income", "Rating"]])
plt.show()

# Unnecessary columns drop karein
df = df.drop(["Invoice ID", "Date", "Time", "cogs", "gross margin percentage"], axis=1)
print(df.head())

# Features (Independent Variables)
X = df.drop(["Branch"], axis=1)  # 'Branch' ko predict karna hai, isliye hata diya
# Target (Dependent Variable)
y = df["Branch"]



from sklearn.preprocessing import LabelEncoder

# Encoding ke liye categorical columns select karein
cat_cols = ["Customer type", "Gender", "Product line", "Payment"]
encoder = LabelEncoder()

# Har categorical column ko encode karein
for col in cat_cols:
 X[col] = encoder.fit_transform(X[col])

print(X.head())  # Check karo encoded data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.dtypes) 

X_train["City"] = encoder.fit_transform(X_train["City"])
X_test["City"] = encoder.transform(X_test["City"])  # Test data bhi encode karo
 # Sabhi columns ka datatype check karo
from sklearn.ensemble import RandomForestClassifier

# Model Define Karna
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Model Train Karna
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Predictions
y_pred = model.predict(X_test)
print(y_pred)
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# # print(df.head())
