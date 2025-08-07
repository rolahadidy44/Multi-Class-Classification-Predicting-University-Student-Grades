# Multi-Class-Classification-Predicting-Student-Grades

This project aims to predict student performance using supervised learning models and explore student groupings through unsupervised clustering techniques.  
We apply classification algorithms to predict grades, and use KMeans to identify meaningful student clusters.

## Technologies Used
 * Python 3
 * Jupyter Notebook
 * Pandas and Numpy
 * Matplotlib and Seaborn
 * Scikit-learn 
 * Git & GitHub

## Project Structure
├── analysis
│   ├── analysis.ipynb
│   └── analysis.py
├── data
│   ├── cleaned_data123.csv
│   ├── cleaned_dataWithOtliers.csv
│   └── StudentPerformanceFactors-1.csv
├── Pipfile
├── Pipfile.lock
├── README.md
├── requirements.txt
├── savedmodels
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── rf_model.pk1
│   └── svm.pkl
├── supervisedlearning
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── gradient_boost.py
│   ├── __init__.py
│   ├── preprocess.py
│   ├── __pycache__
│   │   ├── data_loader.cpython-310.pyc
│   │   ├── evaluate.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   ├── preprocess.cpython-310.pyc
│   │   ├── savemodels.cpython-310.pyc
│   │   └── train.cpython-310.pyc
│   ├── savemodels.py
│   ├── train.ipynb
│   ├── train.py
│   └── utils.py
└── unsupervised
    ├── kmeans.py
    └── unsupervise.ipynb

## Installment Instructions
 * Clone the repo
    git clone https://github.com/rolahadidy44/Multi-Class-Classification-Predicting-University-Student-Grades

* Install the required libraries 
    pip install -r requirements.txt