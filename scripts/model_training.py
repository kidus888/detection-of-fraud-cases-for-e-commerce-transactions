import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow

# Data Preparation: Feature and Target Separation
def prepare_data(df, target_column):
    """Separate features and target variable."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# Train-Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model Selection and Training
def train_model(model, X_train, X_test, y_train, y_test):
    """Train the model and evaluate performance."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)
    
    return accuracy, conf_matrix, report

# Experiment Tracking with MLflow
def log_experiment(model_name, accuracy, conf_matrix, report, params):
    """Log experiments in MLflow."""
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("parameters", params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact("confusion_matrix.txt", conf_matrix)
        mlflow.log_artifact("classification_report.txt", report)


