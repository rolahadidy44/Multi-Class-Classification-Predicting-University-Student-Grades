from preprocess import preprocess
from savemodels import save_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

# so we can import from root-level
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supervisedlearning.data_loader import load_data

# Loading
df = load_data()
x_train_balanced, x_test_scaled, y_train_balanced, y_test = preprocess(df)

# Train 
model = GradientBoostingClassifier()
model.fit(x_train_balanced, y_train_balanced)
y_pred = model.predict(x_test_scaled)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("\nGradient Boosting")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Gradient Boosting - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Save the model
save_model(model, "Gradient_Boosting")
