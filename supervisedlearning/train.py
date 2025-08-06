from preprocess import preprocess
from evaluate import evaluate_models
from savemodels import save_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supervisedlearning.data_loader import load_data

df = load_data()
x_train_balanced, x_test_scaled, y_train_balanced, y_test = preprocess(df)

def train_models(x_train_balanced, x_test_scaled, y_train_balanced, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train_balanced, y_train_balanced )
        save_model(model, name) 
        y_pred = model.predict(x_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))


    return models
trained_models = train_models(x_train_balanced, x_test_scaled, y_train_balanced, y_test)
evaluate_models(trained_models, x_test_scaled, y_test)


