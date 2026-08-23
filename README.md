# Supply Chain Analytics & Customer Segmentation

## Project Overview

This project analyzes 180K+ supply chain transactions from the DataCo Supply Chain Dataset to understand sales performance, profitability, and customer behavior. It combines exploratory data analysis (EDA), RFM-based feature engineering, unsupervised customer segmentation, and a supervised classification model to translate raw transactional data into actionable business recommendations.

---

## Business Problem

Not all customers, products, or categories contribute equally to profitability. The goals of this project are to:

* Understand where sales and profit are actually being generated (by category, market, product, and discount level).
* Identify distinct customer segments based on purchasing behavior, spend, and profitability.
* Build a model that can classify customers into these segments using their behavioral data.
* Turn these findings into concrete recommendations for retention, discount strategy, and profitability optimization.

---

## Dataset Information

**Dataset:** DataCo Supply Chain Dataset

**Records:** 180,519 order-level transactions

### Key Fields Used

* Sales, Order Profit Per Order, Order Item Profit Ratio
* Order Item Discount, Order Item Discount Rate
* Category Name, Department Name, Product Name
* Market, Order Country, Order Region
* Customer Id, Customer Segment
* Order Date

### Engineered Features

* Discount Bucket (binned discount rate)
* Recency, Frequency, Monetary (RFM)
* Average Order Value
* Unique Product Categories per customer
* Average Discount Rate per customer
* Total Profit per customer

---

## Tools & Technologies

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn (StandardScaler, KMeans, RandomForestClassifier)
* Jupyter Notebook
* joblib (model persistence)

---

## Part 1: Sales & Profitability Analysis

* Total Sales, Total Profit, Profit Margin %, Average Order Value
* Monthly sales and profit trends by year
* Top/bottom 10 products by sales and profit
* Sales and profit by category and by market
* Profit margin % by category
* Discount rate vs. average profit (bucketed analysis)

**Key Findings**
* Sales volume and profitability don't always align — some categories/products sell well but contribute little margin.
* Higher discount rates are associated with lower average profit per order.
* Profitability is concentrated in a subset of categories and products rather than spread evenly.

---

## Part 2: RFM Analysis & Customer Segmentation

Built a per-customer RFM table (Recency, Frequency, Monetary) and extended it with:

* Average Order Value
* Unique Product Categories purchased
* Average Discount Rate
* Total Profit generated

Standardized these features and applied **K-Means clustering**, using the elbow method to select **K = 5** clusters across ~20,500 customers.

### Segment Profiles

| Segment | Recency (days) | Frequency | Monetary | Avg Discount | Categories | Total Profit | % of Customers | % of Total Profit |
|---|---|---|---|---|---|---|---|---|
| High-Value / Champions | 264 | 4.29 | $2,474 | 10% | 4.48 | $633.61 | 22.3% | **73.3%** |
| At-Risk / Lapsed | 559 | 2.77 | $1,557 | 10% | 4.05 | $151.34 | 15.0% | 11.7% |
| New / Low-Value | 67 | 1.07 | $342 | 5% | 1.07 | $32.35 | 23.2% | 3.9% |
| High-Volume, Thin-Margin | 240 | 6.49 | $4,054 | 10% | 3.90 | $76.60 | 21.0% | 8.3% |
| Discount-Driven | 66 | 1.05 | $334 | 17% | 1.06 | $28.60 | 18.6% | 2.75% |

**Key Finding:** The High-Value segment is only 22.3% of customers but generates 73.3% of total profit — a strong Pareto-style concentration that should drive retention priorities.

---

## Part 3: Customer Segment Classification Model

### Objective
Build a supervised model that classifies a customer into one of the 5 segments based on their behavioral features — enabling fast segment assignment for new/incoming customers without re-running clustering on the full dataset.

### Model Used
Random Forest Classifier (200 estimators)

### Features
Recency, Frequency, Monetary, Avg Discount Rate, Unique Product Categories, Total Profit

### Model Performance

| Metric | Score |
|---|---|
| Accuracy | 97.5% |
| Macro F1 | 0.97 |
| Weighted F1 | 0.98 |

### Top Feature Importances
1. Avg Discount Rate (26.2%)
2. Recency (21.0%)
3. Total Profit (16.8%)
4. Frequency (13.5%)
5. Monetary (12.5%)
6. Unique Product Categories (10.1%)

Discount behavior and recency are the strongest drivers of segment membership — more predictive than raw spend alone.

---

## Business Recommendations

* **High-Value segment (73% of profit):** Prioritize retention — loyalty programs, dedicated account management, early-warning monitoring for rising Recency.
* **At-Risk/Lapsed segment:** Targeted win-back campaigns before they fully churn.
* **New/Low-Value segment:** Onboarding and cross-sell campaigns to increase category diversity and frequency.
* **High-Volume, Thin-Margin segment:** Review discount depth and product mix — high revenue isn't translating into proportional profit.
* **Discount-Driven segment:** Reassess acquisition cost and promotional strategy — this is the least profitable segment per customer.

---

## Repository Structure

```text
Supply-Chain-Analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── Sales & Profitability Analysis/
│   ├── data_preprocessing.ipynb
│   └── profit_sales_analysis.ipynb
│
├── Customer-Segmentation/
│   ├── rfm_analysis.ipynb
│   ├── ml_model.ipynb
│   └── models/
│
├── README.md
└── requirements.txt
```

---

## Future Work

* Cross-validation and hyperparameter tuning for the classifier
* Silhouette score analysis to further validate cluster quality
* Incorporate delivery/shipping performance fields (Delivery Status, Late Delivery Risk) for a delivery-risk analysis module
* Build a live dashboard for ongoing segment monitoring

---

## Author

Rupjit Das
