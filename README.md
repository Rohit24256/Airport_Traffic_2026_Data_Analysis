# ✈️ Airport Traffic 2026 Data Analysis

## 📌 Project Overview

This project focuses on analyzing airport traffic data for **January 2026** to uncover meaningful insights into flight operations. The dataset includes information on **departures, arrivals, total flights, IFR traffic, airport names, and country/state distribution**.

The main objective is to perform **Exploratory Data Analysis (EDA)**, detect anomalies, visualize patterns, and build predictive models to understand air traffic behavior.

---

## 🎯 Objectives

* Analyze flight traffic trends over time
* Identify top-performing airports and regions
* Detect outliers in flight data
* Understand relationships between key variables
* Build predictive models for total flight estimation
* Perform statistical testing on flight patterns

---

## 📂 Dataset Description

The dataset contains the following key features:

* **FLT_DATE** → Date of flight operations
* **FLT_DEP_1** → Total departures
* **FLT_ARR_1** → Total arrivals
* **FLT_TOT_1** → Total flights (departures + arrivals)
* **FLT_DEP_IFR_2 / FLT_ARR_IFR_2 / FLT_TOT_IFR_2** → IFR flight metrics
* **APT_NAME** → Airport name
* **STATE_NAME** → Country/State

---

## 🧹 Data Cleaning & Preprocessing

* Converted `FLT_DATE` to datetime format
* Removed missing or invalid values
* Filtered dataset for consistent analysis
* Created separate dataset for state-level analysis
* Verified dataset integrity using summary statistics

---

## 📊 Exploratory Data Analysis (EDA)

### ✔️ Key Analysis Performed

* Dataset overview (shape, info, statistics)
* Missing value analysis
* Correlation matrix between flight variables
* Distribution analysis using histograms
* Outlier detection using **IQR method**

---

## 📈 Visualizations

* 📅 Daily flight traffic trends
* 🌍 Top countries/states by total flights
* 🛫 Top airports by IFR traffic
* 📊 Correlation heatmap
* 📉 Distribution plots and boxplots

---

## 🤖 Machine Learning Models

### 🔹 Linear Regression Models

1. **Departures → Total Flights**
2. **Arrivals → Total Flights**

### 📌 Evaluation Metrics

* Mean Squared Error (MSE)
* R² Score

These models help estimate total flights based on operational inputs.

---

## 📊 Statistical Analysis

### 🔬 Hypothesis Testing

* Compared **weekday vs weekend flight traffic**
* Used **Two-sample t-test**

**Hypothesis:**

* H0: No difference in mean traffic
* H1: Significant difference exists

---

## 🔍 Key Insights

* Strong correlation between departures, arrivals, and total flights
* Certain airports dominate overall and IFR traffic
* Flight activity shows noticeable daily fluctuations
* Outliers indicate unusual spikes in traffic
* Predictive models provide reasonable accuracy
* Weekday vs weekend traffic patterns can differ

---

## 🛠️ Technologies Used

* **Python**
* **Pandas & NumPy**
* **Matplotlib & Seaborn**
* **Scikit-learn**
* **SciPy**

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Airport_Traffic_2026_Data_Analysis.git
cd Airport_Traffic_2026_Data_Analysis
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3️⃣ Run the Script

```bash
python airport_traffic_analysis.py
```

---

## 📁 Project Structure

```
Airport_Traffic_2026_Data_Analysis/
│
├── airport_traffic_analysis.py
├── airport_traffic_2026.csv
├── README.md
```

---

## ✅ Conclusion

This project demonstrates how data analytics techniques can be applied to aviation datasets to extract actionable insights. By combining **EDA, visualization, machine learning, and statistical analysis**, we gain a deeper understanding of airport traffic patterns.

---

## 🔮 Future Scope

* Implement advanced ML models (Random Forest, XGBoost)
* Build interactive dashboards (Power BI / Tableau)
* Extend analysis across multiple months/years
* Integrate weather and delay datasets
* Develop real-time traffic monitoring system

---

## 🙌 Acknowledgements

This project was developed as part of a data analysis learning initiative to enhance practical skills in Python, visualization, and machine learning.
