#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import numpy as np
import os


# In[2]:
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "StudentPerformanceFactors-1.csv")
df= pd.read_csv(data_path)


# In[3]:


df.head()


# In[4]:


col_names=df.columns.tolist()

for col in col_names:
    null_count=df[col].isnull().sum()
    print(col,": ",null_count)


# In[114]:


print("number of rows: ",len(df))


# In[115]:


print(df.dtypes)


# In[116]:


for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
    if df[col].isnull().sum() > 0:
        most_frequent = df[col].mode()[0]
        df[col].fillna(most_frequent, inplace=True)
        print(f"Filled missing values in '{col} with: {most_frequent}")
        


# In[117]:


print(df.isnull().sum())


# In[118]:


df.describe()


# In[119]:


def score_to_grade(score):
    if pd.isna(score):
        return None
    elif score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['Previous_Scores'] = df['Previous_Scores'].apply(score_to_grade)
df['Exam_Score'] = df['Exam_Score'].apply(score_to_grade)

df[['Previous_Scores','Exam_Score']].head()    


# In[120]:


df= pd.read_csv("../data/StudentPerformanceFactors-1.csv")


# In[121]:


#label encoding

mapping={
    'Parental_Involvement': {'Low': 1, 'Medium':2, 'High':3},
    'Access_to_Resources':{'Low': 1, 'Medium':2, 'High':3},
    'Motivation_Level':{'Low': 1, 'Medium':2, 'High':3},
    'Family_Income':{'Low': 1, 'Medium':2, 'High':3},
    'Teacher_Quality': {'Low': 1, 'Medium': 2, 'High': 3},
    'Distance_from_Home': {'Near': 1, 'Moderate': 2, 'Far': 3},
    'Parental_Education_Level': {'High School': 1, 'College': 2, 'Postgraduate': 3},
    'Peer_Influence': {'Negative': 1, 'Neutral': 2, 'Positive': 3}
}

for col,mapping in mapping.items():
    df[f'{col}_label'] = df[col]   # saving the original col for plotting purposes
    df[col]=df[col].map(mapping)
    

df.head()


# In[122]:


#one hot endcoding
nominal_cols=[
    'Extracurricular_Activities',
    'Internet_Access',
    'Learning_Disabilities',
    'School_Type',
    'Gender'   
]

for col in nominal_cols:
    df[f'{col}_label'] = df[col]  # for plotting purposes,,again

df=pd.get_dummies(df, columns=nominal_cols, drop_first=False)

df.head()


# In[123]:


def score_to_grade(score):
    if pd.isna(score):
        return None
    elif score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

# df['Previous_Scores'] = df['Previous_Scores'].apply(score_to_grade)
df['Exam_Score'] = df['Exam_Score'].apply(score_to_grade)

df[['Previous_Scores','Exam_Score']].head()    
df.head()



# In[124]:


# correlation matrix

df_numeric=  df.select_dtypes(include=['number'])
correlation_matrix = df_numeric.corr()    
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})  
plt.title('Correlation Matrix')
plt.show()  


# In[125]:


# Histogram of Previous_Scores

sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.countplot(x = 'Previous_Scores', data=df, order=['A', 'B', 'C', 'D', 'F'])
plt.title("Distribution of prevoius scores (Grades)")
plt.xlabel("Grade")
plt.ylabel("Count")
plt.show()


# In[126]:


sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.countplot(x = 'Exam_Score', data=df, order=['A', 'B', 'C', 'D', 'F'])
plt.title("Distribution of Final Exam Scores (Grades)")
plt.xlabel("Grade")
plt.ylabel("Count")
plt.show()


# In[127]:


cleaned_file = df.copy()

for column in cleaned_file.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = cleaned_file[column].quantile(0.25)
    Q3 = cleaned_file[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5* IQR
    cleaned_file = cleaned_file[(cleaned_file[column] >= lower) & (cleaned_file[column] <= upper)]
    
cleaned_file.to_csv('../data/cleaned_data123.csv', index=False)  #new cleanded file
    


# In[128]:


print("Original rows:", df.shape[0])
print("Rows after outlier removal:", cleaned_file.shape[0])
print("Rows removed:", df.shape[0] - cleaned_file.shape[0])


# In[129]:


cleaned_df = pd.read_csv("../data/cleaned_data123.csv")
sns.set(style="whitegrid")

sns.set(style="whitegrid")
plt.figure(figsize=(6,4))
sns.countplot(x='Exam_Score', data=cleaned_df, order=['A', 'B', 'C', 'D', 'F'])
plt.title("Distribution of Final Exam Scores (Grades)")
plt.xlabel("Grade")
plt.ylabel("Count")
plt.show()


# In[130]:


print("number of rows: ",len(cleaned_df))


# In[131]:


print(df['Exam_Score'].value_counts())             # Before outlier removal
print(cleaned_df['Exam_Score'].value_counts()) 


# In[132]:


col_names=df.columns.tolist()

for col in col_names:
    print(col)


# In[133]:


sns.histplot(df["Sleep_Hours"], kde=True)


# In[134]:


sns.boxplot(data=df, x="Gender_label", y="Exam_Score")


# In[135]:


sns.barplot(data=df, x="Internet_Access_label", y="Exam_Score")


# In[136]:


sns.pairplot(df, vars=["Hours_Studied", "Previous_Scores", "Exam_Score", "Sleep_Hours"])

