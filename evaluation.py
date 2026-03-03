import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from preprocessing import load_data
import numpy as np

X_train, X_test, y_train, y_test = load_data("data/creditcard.csv")

model = joblib.load("models/LightGBM.pkl")

y_probs = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Example cost assumptions
cost_false_negative = 500   # missed fraud cost
cost_false_positive = 5     # blocking legit user cost

tn, fp, fn, tp = cm.ravel()

total_loss = (fn * cost_false_negative) + (fp * cost_false_positive)

print("Estimated Financial Loss: $", total_loss)