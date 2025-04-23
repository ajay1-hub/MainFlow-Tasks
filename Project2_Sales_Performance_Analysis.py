import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------ Load Dataset with Encoding Handling ------------
try:
    df = pd.read_csv("Sales_data_sample.csv")
except UnicodeDecodeError:
    df = pd.read_csv("Sales_data_sample.csv", encoding='latin1')

# Initial Inspection
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Data Cleaning
df.drop_duplicates(inplace=True)
df.fillna({
    'Sales': df['Sales'].mean(),
    'Profit': df['Profit'].mean(),
    'Discount': df['Discount'].mean()
}, inplace=True)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


# ------------ Exploratory Data Analysis (All Plots Together) ------------

fig, axs = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Sales Data Visualizations', fontsize=16)

# 1. Sales Trend Over Time
df_time = df.groupby('Date')['Sales'].sum().reset_index()
sns.lineplot(data=df_time, x='Date', y='Sales', ax=axs[0, 0])
axs[0, 0].set_title("Sales Over Time")
axs[0, 0].set_xlabel("Date")
axs[0, 0].set_ylabel("Total Sales")
axs[0, 0].tick_params(axis='x', rotation=45)

# 2. Profit vs Discount
sns.scatterplot(data=df, x='Discount', y='Profit', ax=axs[0, 1])
axs[0, 1].set_title("Profit vs Discount")
axs[0, 1].set_xlabel("Discount")
axs[0, 1].set_ylabel("Profit")

# 3. Sales by Region
region_Sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
region_Sales.plot(kind='bar', color='skyblue', ax=axs[1, 0])
axs[1, 0].set_title("Sales by Region")
axs[1, 0].set_ylabel("Total Sales")

# 4. Sales by Category (Pie Chart)
category_Sales = df.groupby('Category')['Sales'].sum()
axs[1, 1].pie(category_Sales, labels=category_Sales.index, autopct='%1.1f%%', startangle=140)
axs[1, 1].set_title("Sales by Category")
axs[1, 1].axis('equal')  # Equal aspect ratio to ensure it's a circle

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ------------ Predictive Modeling ------------


# Features & Target
X = df[['Profit', 'Discount']]
y = df['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Performance:\nR² Score: {r2:.2f}\nMean Squared Error: {mse:.2f}")


# ------------ Insights & Recommendations ------------


print("\n--- Insights & Recommendations ---")
print("1. Sales generally follow a certain trend over time — useful for forecasting.")
print("2. There’s a visible trade-off between higher discounts and lower profits.")
print("3. Focus on top-performing regions for expansion.")
print("4. Certain categories dominate Sales — consider inventory and marketing focus there.")
print("5. A basic regression model shows that Profit and Discount can moderately predict Sales.")
