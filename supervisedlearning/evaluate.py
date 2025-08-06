from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_models(models, x_test, y_test):
    for name, model in models.items():
        print(f"\n==== {name} ====")
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
        
         # Feature Importance (if available)
    if hasattr(model, "feature_importances_"):
        print(f"[INFO] Plotting Feature Importance for {name}...")
        importances = model.feature_importances_

        # Use proper column names if available
        try:
            feature_names = x_test.columns
        except AttributeError:
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        features = pd.Series(importances, index=feature_names)
        features.sort_values().plot(kind='barh', figsize=(10, 6))
        plt.title(f"{name} - Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show(block=True)