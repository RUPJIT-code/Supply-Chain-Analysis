# Supply Chain Delivery Performance Analysis & Late Delivery Risk Prediction

## Project Overview

This project analyzes supply chain delivery performance using the DataCo Supply Chain Dataset. The objective is to evaluate delivery efficiency, identify factors contributing to delayed shipments, and build a machine learning model to predict late-delivery risk.

The project combines exploratory data analysis (EDA), business intelligence reporting, and predictive analytics to provide actionable insights for logistics and supply chain decision-making.

---

## Business Problem

Delayed deliveries negatively impact customer satisfaction, operational efficiency, and logistics costs.

The goal of this project is to:

* Measure delivery performance across the supply chain.
* Identify shipping modes and regions with higher delay rates.
* Analyze delivery trends and operational bottlenecks.
* Predict whether an order is likely to be delivered late.

---

## Dataset Information

**Dataset:** DataCo Supply Chain Dataset

**Records:** 180,519 Orders

### Key Variables

* Delivery Status
* Late Delivery Risk
* Days for Shipping (Real)
* Days for Shipment (Scheduled)
* Shipping Mode
* Order Region
* Order Country
* Order Date

### Derived Features

* Delay Days
* Delay Level
* Order Month

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## Analysis Performed

### Delivery Performance Analysis

* Delivery Status Distribution
* Late Delivery Risk Analysis
* Actual vs Scheduled Shipping Days
* Average Delay by Shipping Mode
* Late Delivery Rate by Shipping Mode
* Regional Delivery Performance
* Country-Level Delivery Analysis
* Monthly Delivery Trends
* Delivery Delay Distribution
* Region × Shipping Mode Heatmap

### KPI Dashboard

* Total Orders
* On-Time Delivery Rate
* Late Delivery Rate
* Average Shipping Days
* Average Delay Days

---

## Machine Learning Model

### Objective

Predict whether an order will experience delivery delays.

### Model Used

Random Forest Classifier

### Features

* Shipping Mode
* Order Region
* Order Country
* Scheduled Shipping Days

### Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 69.37% |
| Precision | 83.30% |
| Recall    | 55.20% |
| F1 Score  | 66.40% |

---

## Key Findings

* Delivery performance varies significantly across shipping modes.
* Certain regions consistently experience higher delay rates.
* Actual shipping duration frequently exceeds scheduled shipping duration.
* Geographic location has a significant impact on delivery reliability.
* Machine learning can effectively identify high-risk shipments before delivery.

---

## Business Recommendations

* Improve logistics operations in high-risk regions.
* Optimize underperforming shipping modes.
* Monitor high-risk shipments proactively.
* Review scheduling practices to reduce delays.
* Leverage predictive analytics to support logistics planning.

---

## Repository Structure

```text
Supply-Chain-Analysis/
│
├── notebooks/
├── reports/
├── visuals/
├── README.md
└── requirements.txt
```

---

## Author

Rupjit Das

