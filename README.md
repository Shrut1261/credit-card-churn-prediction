#  Credit Card Churn Prediction

##  Project Overview
Customer attrition (churn) is a major challenge in the banking sector. Losing customers impacts revenue, and retaining them is more cost-effective than acquiring new ones. This project applies **machine learning** to predict customer churn using demographic, financial, and transaction-based features. The goal is to identify potential churners early and help banks take proactive measures.

##  Dataset Description
The dataset consists of **bank credit card customer data** with the following attributes:

- **Demographic Information**: Age, Gender, Marital Status, etc.
- **Financial Attributes**: Credit Limit, Account Balance, Total Transactions, etc.
- **Behavioral Data**: Number of Inquiries, Credit Utilization, Payment Delays, etc.
- **Target Variable**: `Attrition_Flag` (1 = Churned, 0 = Retained)

### ** Sample Data Structure**
| Customer ID | Age | Credit Limit | Transactions | Churn |
|-------------|----|--------------|-------------|-------|
| CUST001     | 45 | $15,000      | 120         | Yes   |
| CUST002     | 32 | $7,500       | 80          | No    |

---

## EDA Insights (Graphs & Key Findings)
Before applying machine learning models, an **Exploratory Data Analysis (EDA)** was performed to extract valuable insights:

### ** Churn Rate Distribution**
- The dataset is imbalanced, with fewer customers churning.
- **Older customers** tend to churn more often.
- Customers with **higher credit utilization** show a higher churn rate.

![Churn Distribution](path/to/churn_distribution.png)

### ** Transaction Frequency vs Churn**
- Customers with **fewer transactions per month** are more likely to churn.
- Engaged customers (high transaction count) have a lower churn rate.

![Transaction Patterns](path/to/transaction_patterns.png)

### ** Feature Importance**
- The most influential features in predicting churn:
  - **Credit Utilization**
  - **Total Transactions**
  - **Customer Age**
  - **Account Balance**

![Feature Importance](path/to/feature_importance.png)

---

## Machine Learning Approach
### ** Data Preprocessing**
- **Handling Missing Values**
- **Encoding Categorical Features**
- **Feature Scaling (Standardization)**

### ** Models Used**
Three machine learning models were trained and evaluated:

| Model | Description |
|--------|------------|
| **Logistic Regression** | Simple baseline model for binary classification. |
| **Gaussian Naïve Bayes** | Probabilistic model handling categorical data. |
| **ARIMA** | Time-series forecasting for churn trends. |

### ** Model Evaluation**
Metrics used for evaluation:
- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**
- **ROC Curve**

---

##  Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 85% | 82% | 80% | 81% |
| Naïve Bayes | 83% | 80% | 78% | 79% |
| ARIMA Forecast | N/A | N/A | N/A | N/A |

### **ROC Curve**
![ROC Curve](path/to/roc_curve.png)

---

## Business Insights & Recommendations
 **Identify High-Risk Customers**: Focus on customers with low transaction activity and high credit utilization.  
 **Offer Personalized Incentives**: Provide loyalty rewards or lower interest rates to retain customers.  
 **Proactive Communication**: Use automated alerts to notify at-risk customers.  
 **Leverage ARIMA Forecasting**: Prepare marketing campaigns in advance for predicted churn periods.  

---

## ⚙️ How to Run the Code
### **Clone the Repository**
```bash
git clone https://github.com/your-username/Credit-Card-Churn-Prediction.git
cd Credit-Card-Churn-Prediction
