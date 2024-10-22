# detection-of-fraud-cases-for-e-commerce-transactions

# Fraud Detection Project

This project focuses on detecting fraudulent transactions using machine learning techniques on two datasets: `Fraud_Data.csv` and `creditcard.csv`. The project consists of two primary tasks: Data Analysis & Preprocessing (Task 1) and Model Building & Training (Task 2).

## Project Structure

The project is organized as follows:


- **data/**: Contains the datasets used for this project.
- **notebooks/**: Contains the Jupyter notebooks used for running the analysis and training models.
- **scripts/**: Contains Python scripts for data preprocessing and model building.

## Datasets

1. **Fraud_Data.csv**: E-commerce transaction data aimed at identifying fraudulent activities.
   - `user_id`: Unique identifier for the user who made the transaction.
   - `signup_time`: The timestamp when the user signed up.
   - `purchase_time`: The timestamp when the purchase was made.
   - `purchase_value`: The value of the purchase in dollars.
   - `device_id`: Unique identifier for the device used to make the transaction.
   - `source`: The source through which the user came to the site (e.g., SEO, Ads).
   - `browser`: The browser used to make the transaction (e.g., Chrome, Safari).
   - `sex`: Gender of the user (M for male, F for female).
   - `age`: Age of the user.
   - `ip_address`: The IP address from which the transaction was made.
   - `class`: Target variable where `1` indicates a fraudulent transaction and `0` indicates a non-fraudulent transaction.

2. **IpAddress_to_Country.csv**: Maps IP addresses to countries.
   - `lower_bound_ip_address`: The lower bound of the IP address range.
   - `upper_bound_ip_address`: The upper bound of the IP address range.
   - `country`: The country corresponding to the IP address range.

3. **creditcard.csv**: Contains bank transaction data specifically curated for fraud detection analysis.
   - `Time`: Time elapsed between the transaction and the first transaction in the dataset.
   - `V1` to `V28`: Anonymized features resulting from PCA transformation.
   - `Amount`: The transaction amount in dollars.
   - `Class`: Target variable where `1` indicates a fraudulent transaction and `0` indicates a non-fraudulent transaction.

## Task 1 - Data Analysis and Preprocessing

In this task, we handle missing values, clean the data, and perform feature engineering. The steps involved are:

1. **Handle Missing Values**: Impute or drop missing values in the datasets.
2. **Data Cleaning**: Remove duplicates, correct data types, and convert `datetime` columns into numerical features.
3. **Exploratory Data Analysis (EDA)**:
   - Univariate analysis of key features such as age, purchase value, and signup-to-purchase time.
   - Bivariate analysis to understand relationships between features and fraud occurrence.
4. **Merge Datasets for Geolocation Analysis**: Convert IP addresses to integer format and merge `Fraud_Data.csv` with `IpAddress_to_Country.csv`.
5. **Feature Engineering**:
   - Create features such as `transaction frequency`, `signup_to_purchase` (time difference), `purchase_hour`, and `purchase_day_of_week`.
6. **Normalization and Scaling**: Scale numerical features and encode categorical features like `device_id`, `source`, and `browser`.

### Data Preprocessing Code

The data preprocessing logic is encapsulated in `scripts/data_preprocessing.py`, and it can be called in your notebooks or other scripts. The code handles missing values, datetime conversion, feature engineering, and scaling.

### Usage

```python
from scripts.data_preprocessing import preprocess_fraud_data, preprocess_creditcard_data

# Preprocess Fraud Data
X_fraud_train_scaled, X_fraud_test_scaled, y_fraud_train, y_fraud_test = preprocess_fraud_data('data/Fraud_Data.csv', 'data/IpAddress_to_Country.csv')

# Preprocess Credit Card Data
X_credit_train_scaled, X_credit_test_scaled, y_credit_train, y_credit_test = preprocess_creditcard_data('data/creditcard.csv')

## Task 2 - Model Building and Training

 In Task 2, we build and train various machine learning models, including:

Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Multi-Layer Perceptron (MLP)
Convolutional Neural Network (CNN)
Recurrent Neural Network (RNN)
Long Short-Term Memory (LSTM)
Model Building and Evaluation
We train models on both the Fraud_Data.csv and creditcard.csv datasets, evaluate them, and track model performance. The code also includes MLOps features like versioning and experiment tracking using MLflow.

Model Training Code
The model training logic is implemented in scripts/model_training.py, which handles the training, evaluation, and tracking of multiple models.



### Key Sections:
1. **Project Structure**: Explains the structure of the project and where to find the relevant files.
2. **Datasets**: Describes the datasets used in the project.
3. **Task 1: Data Analysis and Preprocessing**: Provides details on how the preprocessing was done, and includes usage instructions for the preprocessing code.
4. **Task 2: Model Building and Training**: Describes the different models and how to train them, along with code usage.
5. **MLOps**: Describes how MLflow is used for tracking and versioning.
6. **Installation**: Steps to set up the project on a local machine.

Feel free to modify the `README.md` file as needed for your project!
