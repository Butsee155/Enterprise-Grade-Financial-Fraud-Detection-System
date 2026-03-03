from preprocessing import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

X_train, X_test, y_train, y_test = load_data("data/creditcard.csv")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "LightGBM": LGBMClassifier(),
    "Isolation Forest": IsolationForest(contamination=0.002)
}

for name, model in models.items():

    if name == "Isolation Forest":
        model.fit(X_train)
        preds = model.predict(X_test)
        preds = [1 if p == -1 else 0 for p in preds]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    print(f"\n{name}")
    print(classification_report(y_test, preds))

    if name != "Isolation Forest":
        print("ROC-AUC:",
              roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")