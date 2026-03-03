import shap
import joblib
import pandas as pd
from preprocessing import load_data

X_train, X_test, y_train, y_test = load_data("data/creditcard.csv")

model = joblib.load("models/LightGBM.pkl")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)