from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def preprocess(df):
    y=df['Exam_Score']
    x=df.drop(['Exam_Score'], axis=1)


    #droppinf the cols which end with _label bec they are unnecessary now
    x = x[[col for col in x.columns if not col.endswith('_label')]]
    
    #splitting the data
    x_train,x_test,y_train,y_test= train_test_split (x,y, test_size=0.3, random_state=42)
    
    #scaling the data
    scaler = StandardScaler()
    x_train_scaled=pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test_scaled=pd.DataFrame(scaler.transform(x_test),columns=x.columns)
    
    #removing non-numerical values
    x_train_scaled = np.nan_to_num(x_train_scaled)
    x_test_scaled = np.nan_to_num(x_test_scaled)
    
    #smote to balance the data
    
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

    # Convert y_train_balanced to a labeled Series
    y_train_balanced = pd.Series(y_train_balanced, name='Exam_Score')

    print("Before SMOTE:\n", y_train.value_counts())
    print("\nAfter SMOTE:\n", y_train_balanced.value_counts())
    
    return x_train_balanced, x_test_scaled, y_train_balanced, y_test