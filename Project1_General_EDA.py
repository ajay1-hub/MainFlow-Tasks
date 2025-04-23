import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: clean output and display
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')

# 1.---------- LOAD DATA ----------
df = pd.read_csv("GlobalSuperstore.csv", encoding='ISO-8859-1')
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
print("Initial shape:", df.shape)


# 2. -------- CLEANING ----------

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Fill numeric missing values
numeric_cols = ['Sales', 'Profit', 'Discount', 'Quantity']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Fill categorical missing values
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Drop any remaining rows with missing values
df.dropna(inplace=True)

# Remove duplicates
print("Duplicates found:", df.duplicated().sum())
df.drop_duplicates(inplace=True)


# 3. ---------- OUTLIER HANDLING (IQR) ----------
def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower) & (dataframe[column] <= upper)]

for col in numeric_cols:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)


# 4. ---------- STATISTICAL ANALYSIS ----------
print("\n--- Descriptive Statistics ---")
for col in numeric_cols:
    if col in df.columns:
        print(f"\n{col} stats:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std Dev: {df[col].std():.2f}")
        print(f"  Variance: {df[col].var():.2f}")

# Correlation matrix
print("\n--- Correlation Matrix ---")
print(df[numeric_cols].corr())


# 5. ---------- VISUALIZATIONS (ALL-IN-ONE) ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram - Sales
df['Sales'].plot.hist(ax=axes[0, 0], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title("Sales Distribution")
axes[0, 0].set_xlabel("Sales")
axes[0, 0].set_ylabel("Frequency")

# Boxplot - Profit
sns.boxplot(x=df['Profit'], ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title("Profit Boxplot")
axes[0, 1].set_xlabel("Profit")

# Barplot - Sales by Region
if 'Region' in df.columns:
    df.groupby('Region')['Sales'].sum().sort_values().plot(kind='bar', ax=axes[1, 0], color='coral')
    axes[1, 0].set_title("Total Sales by Region")
    axes[1, 0].set_ylabel("Sales")
    axes[1, 0].set_xlabel("Region")

# Correlation Heatmap
sns.heatmap(df[numeric_cols].corr(), ax=axes[1, 1], annot=True, cmap='coolwarm', fmt=".2f")
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# Additional: Sales by Category (Separate Plot)
if 'Category' in df.columns:
    plt.figure(figsize=(7,4))
    df.groupby('Category')['Sales'].sum().sort_values().plot(kind='bar', color='lightblue')
    plt.title("Total Sales by Category")
    plt.ylabel("Total Sales")
    plt.xlabel("Category")
    plt.tight_layout()
    plt.show()


# 6. ---------- SUMMARY REPORT ----------
print("\n--- EDA Summary ---")
print(f"Total Records: {len(df)}")
if 'Region' in df.columns:
    print("Most Profitable Region:", df.groupby('Region')['Profit'].sum().idxmax())
if 'Category' in df.columns:
    print("Top-Selling Category:", df.groupby('Category')['Sales'].sum().idxmax())
if 'Segment' in df.columns:
    print("Best Performing Segment:", df.groupby('Segment')['Profit'].sum().idxmax())


# 7. ---------- EXPORT CLEANED DATASET ----------
df.to_csv("Cleaned_GlobalSuperstore.csv", index=False)
print("\nCleaned dataset saved as 'Cleaned_GlobalSuperstore.csv'")
