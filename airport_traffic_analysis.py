import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

df = pd.read_csv("airport_traffic_2026.csv")
print("Shape:")
print(df.shape)
print("Columns:")
print(df.columns)
print("Info:")
print(df.info())
print("Description:")
print(df.describe())
print("First 5 rows:\n")
print(df.head())
print("Last 5 rows:\n")
print(df.tail())

# ─────────────────────────────────────────────
# EDA — DATA CLEANING
# ─────────────────────────────────────────────

# Checking missing values
print("Missing values:\n", df.isnull().sum())

# Creating a working copy
df_clean = df.copy()

# Convert FLT_DATE to datetime
df_clean['FLT_DATE'] = pd.to_datetime(df_clean['FLT_DATE'], errors='coerce')
dropna_count = df_clean['FLT_DATE'].isnull().sum()
print(f"Missing FLT_DATE after conversion: {dropna_count}")
df_clean = df_clean.dropna(subset=['FLT_DATE'])

# Drop rows where key flight count columns are missing
df_clean = df_clean.dropna(subset=['FLT_DEP_1', 'FLT_ARR_1', 'FLT_TOT_1'])

# Dataset with known state (country equivalent)
df_state = df_clean.dropna(subset=['STATE_NAME']).copy()

# Dataset sizes after cleaning
print("\nAfter cleaning:")
print("Full dataset :", df_clean.shape)
print("With state   :", df_state.shape)

# Summary statistics for key variables
print("\nSummary Statistics:")
print(df_clean[['FLT_DEP_1', 'FLT_ARR_1', 'FLT_TOT_1']].describe())

# ─────────────────────────────────────────────
# CORRELATION
# ─────────────────────────────────────────────
numeric_cols = ['FLT_DEP_1', 'FLT_ARR_1', 'FLT_TOT_1', 'FLT_DEP_IFR_2', 'FLT_ARR_IFR_2', 'FLT_TOT_IFR_2']
corr_data = df_clean[numeric_cols].dropna()
corr = corr_data.corr()
print("\nCorrelation Matrix:")
print(corr)

# Heatmap
plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# OUTLIER DETECTION — FLT_TOT_1
# ─────────────────────────────────────────────

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['FLT_TOT_1'], bins=50, kde=True, color='mediumpurple')
plt.title("Total Flight Distribution (VFR+IFR)")
plt.xlabel("Total Flights (FLT_TOT_1)")
plt.ylabel("Frequency")
plt.show()

# Box plot
plt.figure(figsize=(10, 4))
bp = plt.boxplot(df_clean['FLT_TOT_1'], vert=False, patch_artist=True)
bp['boxes'][0].set_facecolor('coral')
bp['medians'][0].set_color('darkred')
plt.title("Outliers in Total Flights (FLT_TOT_1)")
plt.xlabel("Total Flights")
plt.show()

# IQR Method
flt_arr = np.array(df_clean['FLT_TOT_1'])
Q1 = np.percentile(flt_arr, 25)
Q3 = np.percentile(flt_arr, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = flt_arr[(flt_arr < lower) | (flt_arr > upper)]
print("Outliers count (IQR):", len(outliers))

# ─────────────────────────────────────────────
# Question-1
# How is total flight traffic distributed across days in January 2026?
# ─────────────────────────────────────────────
daily_traffic = df_clean.groupby('FLT_DATE')['FLT_TOT_1'].sum().reset_index()

plt.figure()
sns.lineplot(x=daily_traffic['FLT_DATE'], y=daily_traffic['FLT_TOT_1'], color='darkorange', linewidth=2.5)
plt.fill_between(daily_traffic['FLT_DATE'], daily_traffic['FLT_TOT_1'], alpha=0.15, color='darkorange')
plt.title("Daily Total Flights — January 2026")
plt.xlabel("Date")
plt.ylabel("Total Flights")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Question-2
# Which countries (states) dominate by total flights?
# ─────────────────────────────────────────────
state_flt = df_state.groupby('STATE_NAME')['FLT_TOT_1'].sum().sort_values()
top_states = state_flt.tail(10)

plt.figure()
colors_q2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_states)))
plt.barh(top_states.index, top_states.values, color=colors_q2)
plt.title("Top Countries by Total Flights")
plt.xlabel("Total Flights")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Question-3
# Which airports have the highest total IFR traffic?
# ─────────────────────────────────────────────
df_ifr = df_clean.dropna(subset=['FLT_TOT_IFR_2'])
airport_ifr = df_ifr.groupby('APT_NAME')['FLT_TOT_IFR_2'].sum().sort_values()
top_airport_ifr = airport_ifr.tail(10)

plt.figure()
colors_q3 = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_airport_ifr)))
plt.barh(top_airport_ifr.index, top_airport_ifr.values, color=colors_q3)
plt.title("Top 10 Airports by Total IFR Flights")
plt.xlabel("Total IFR Flights")
plt.ylabel("Airport")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Question-4
# Predict total flights (FLT_TOT_1) from departures (FLT_DEP_1)
# ─────────────────────────────────────────────
X = df_clean[['FLT_DEP_1']].values
y = df_clean['FLT_TOT_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("Departures -> Total Flights")
print("MSE     :", mse)
print("R2 Score:", r2)
print(f"Intercept  : {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

plt.figure()
plt.scatter(X, y, color='steelblue', label='Actual Data', alpha=0.4)
plt.plot(X, model.predict(X), color='crimson', linewidth=2, label='Regression Line')
plt.xlabel("Departures (FLT_DEP_1)")
plt.ylabel("Total Flights (FLT_TOT_1)")
plt.title("Linear Regression: Departures vs Total Flights")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Prediction for a new value
new_value = np.array([[500]])
predicted = model.predict(new_value)
print("Predicted total flights for 500 departures:", predicted)

# ── Predict FLT_TOT_1 from arrivals (FLT_ARR_1) ──
X = df_clean[['FLT_ARR_1']].values
y = df_clean['FLT_TOT_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

mse2 = mean_squared_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

print("\nArrivals -> Total Flights")
print("MSE     :", mse2)
print("R2 Score:", r2_2)
print(f"Intercept  : {model2.intercept_:.2f}")
print(f"Coefficient: {model2.coef_[0]:.2f}")

plt.figure()
plt.scatter(X, y, color='mediumseagreen', label='Actual Data', alpha=0.4)
plt.plot(X, model2.predict(X), color='darkorange', linewidth=2, label='Regression Line')
plt.xlabel("Arrivals (FLT_ARR_1)")
plt.ylabel("Total Flights (FLT_TOT_1)")
plt.title("Linear Regression: Arrivals vs Total Flights")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Question-5
# Compare flight traffic: weekdays vs weekends
# H0: Mean flights on weekdays = Mean flights on weekends
# H1: Mean flights on weekdays ≠ Mean flights on weekends
# ─────────────────────────────────────────────
alpha = 0.05

df_clean['day_of_week'] = df_clean['FLT_DATE'].dt.dayofweek   # Mon=0, Sun=6

group_weekday = df_clean[df_clean['day_of_week'] < 5]['FLT_TOT_1']
group_weekend = df_clean[df_clean['day_of_week'] >= 5]['FLT_TOT_1']

t_stat, p_value = stats.ttest_ind(group_weekday, group_weekend, equal_var=True)

print("\nTwo-sample t-test (Weekdays vs Weekends):")
print(f"T-statistic = {t_stat:.4f}")
print(f"P-value     = {p_value:.4f}")

if p_value < alpha:
    print("Conclusion: Reject the null hypothesis.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")

# ─────────────────────────────────────────────
# Question-6
# Top 10 airports by departures, arrivals, and total flights
# ─────────────────────────────────────────────

# Top 10 by Total Flights
top_tot = df_clean.groupby('APT_NAME')['FLT_TOT_1'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
colors_tot = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_tot)))[::-1]
plt.barh(top_tot.index, top_tot.values, color=colors_tot)
plt.gca().invert_yaxis()
plt.title("Top 10 Airports by Total Flights")
plt.xlabel("Total Flights")
plt.ylabel("Airport")
plt.tight_layout()
plt.show()

# Top 10 by Departures
top_dep = df_clean.groupby('APT_NAME')['FLT_DEP_1'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
colors_dep = plt.cm.Purples(np.linspace(0.4, 0.9, len(top_dep)))[::-1]
plt.barh(top_dep.index, top_dep.values, color=colors_dep)
plt.gca().invert_yaxis()
plt.title("Top 10 Airports by Departures")
plt.xlabel("Departures")
plt.ylabel("Airport")
plt.tight_layout()
plt.show()

# Top 10 by Arrivals
top_arr = df_clean.groupby('APT_NAME')['FLT_ARR_1'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
colors_arr = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_arr)))[::-1]
plt.barh(top_arr.index, top_arr.values, color=colors_arr)
plt.gca().invert_yaxis()
plt.title("Top 10 Airports by Arrivals")
plt.xlabel("Arrivals")
plt.ylabel("Airport")
plt.tight_layout()
plt.show()


