# Importing necessary libraries
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def shap_explain_model(model, X_train, X_test, sample_size=100):
    """
    Explains a model using SHAP and creates summary, force, and dependence plots.
    Parameters:
        model: Trained model to explain
        X_train: Training features
        X_test: Test features
        sample_size: Number of samples to use for SHAP explanation
    """
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.KernelExplainer(model.predict, X_train.sample(sample_size))
    shap_values = explainer.shap_values(X_test.sample(sample_size))
    
    # SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test.sample(sample_size), plot_type="bar", show=False)
    plt.title("SHAP Summary Plot")
    plt.show()

    # SHAP Force Plot (single prediction)
    shap.initjs()
    plt.figure()
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.sample(sample_size).iloc[0, :], matplotlib=True)
    plt.title("SHAP Force Plot")
    plt.show()

    # SHAP Dependence Plot
    plt.figure()
    feature = X_test.columns[0]  # Example feature; adjust based on dataset
    shap.dependence_plot(feature, shap_values[1], X_test, show=False)
    plt.title(f"SHAP Dependence Plot for Feature: {feature}")
    plt.show()


def lime_explain_instance(model, X_train, X_test, index=0):
    """
    Explains a single prediction using LIME and creates a feature importance plot.
    Parameters:
        model: Trained model to explain
        X_train: Training features
        X_test: Test features
        index: Index of the test sample to explain
    """
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    
    # Explain a single instance
    exp = explainer.explain_instance(X_test.iloc[index], model.predict_proba)
    
    # LIME Feature Importance Plot
    plt.figure()
    exp.as_pyplot_figure()
    plt.title("LIME Feature Importance for Single Prediction")
    plt.show()
