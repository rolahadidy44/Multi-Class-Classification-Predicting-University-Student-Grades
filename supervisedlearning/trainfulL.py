#!/usr/bin/env python
# coding: utf-8

# In[163]:


# 1. Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib


# In[164]:


# Get current directory of train.py
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "cleaned_data123.csv")

# Load data
df = pd.read_csv(data_path)
#Exam_Score


# In[165]:


#splitting train vs test

y=df['Exam_Score']
x=df.drop(['Exam_Score'], axis=1)


#droppinf the cols which end with _label bec they are unnecessary now
x = x[[col for col in x.columns if not col.endswith('_label')]]


# In[166]:


x_train,x_test,y_train,y_test= train_test_split (x,y, test_size=0.3, random_state=42)


# In[167]:


scaler = StandardScaler()
x_train_scaled=pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
x_test_scaled=pd.DataFrame(scaler.transform(x_test),columns=x.columns)


# In[168]:


x_train_scaled = np.nan_to_num(x_train_scaled)
#
x_test_scaled = np.nan_to_num(x_test_scaled)


# In[169]:





smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

# Convert y_train_balanced to a labeled Series
y_train_balanced = pd.Series(y_train_balanced, name='Exam_Score')

print("Before SMOTE:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", y_train_balanced.value_counts())


# In[170]:


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
    
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))



# In[171]:


plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[172]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
importances = model.feature_importances_
features = pd.Series(importances, index=x_train.columns)
features.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Feature Importance")
plt.show()


# In[173]:


for name, model in models.items():
    model.fit(x_train_balanced, y_train_balanced)
    
    # Training performance
    y_train_pred = model.predict(x_train_balanced)
    train_acc = accuracy_score(y_train_balanced, y_train_pred)

    # Test performance
    y_test_pred = model.predict(x_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"{name}")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print()


# In[174]:


# save a model to a file

joblib.dump(model, "../savedmodels/rf_model.pk1")


# In[175]:


loaded_model = joblib.load("../savedmodels/rf_model.pk1")

