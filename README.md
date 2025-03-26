# supermarket-sales-pridictor-
Objective:
The goal of this project is to analyze, clean, and preprocess the Supermarket Sales dataset and train a machine learning model to predict total sales based on various factors like product type, customer type, branch location, and payment method.

🔹 Step 1: Data Loading
We first loaded the dataset, which contains details about supermarket transactions, including customer details, product categories, sales amounts, and payment methods.

The dataset includes categorical (e.g., gender, product line, payment type) and numerical (e.g., unit price, quantity, total sales) variables.

🔹 Step 2: Data Cleaning & Preprocessing
To ensure high-quality data, we performed several preprocessing steps:

✅ Handling Missing Values
We checked for missing values and removed or imputed them where necessary.

✅ Removing Unnecessary Columns
Some columns, like Invoice ID, Date, and Time, were irrelevant for model training, so we removed them.

✅ Detecting and Removing Outliers
We used the Interquartile Range (IQR) method to identify and remove extreme outliers in numerical features (e.g., unit price, quantity, and total sales).

This helped in improving model performance by removing anomalies.

✅ Converting Categorical Data into Numerical Form
Machine learning models work with numbers, so we converted categorical variables (like Gender, City, Payment Type) into numerical values using Label Encoding.

🔹 Step 3: Splitting Data for Training & Testing
We divided the dataset into features (X) and target (y).

Our target variable was Total Sales, and all other columns were used as features.

We split the data into 80% training and 20% testing sets to train and evaluate the model.

🔹 Step 4: Training the Model
We used Random Forest Regressor, a powerful ensemble learning algorithm, to predict Total Sales.

The model was trained on the training dataset using historical data.

🔹 Step 5: Model Evaluation
We evaluated the model using:

R² Score: Measures how well the model explains the variance in total sales.

Mean Absolute Error (MAE): Measures the average difference between actual and predicted sales.

These metrics helped us understand the accuracy of our predictions.

📌 Conclusion:
This model can help supermarkets predict future sales based on different factors like product type, branch location, and customer behavior.

The insights from the model can assist in inventory management, pricing strategies, and targeted marketing.

📌 Future Improvements:
Feature Engineering: Adding more derived features like average spending per customer.

Hyperparameter Tuning: Optimizing model parameters for better accuracy.

Using Deep Learning: Exploring neural networks for better prediction.
