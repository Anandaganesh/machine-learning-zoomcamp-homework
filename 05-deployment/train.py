#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for an updated module 5 of ML Zoomcamp
# 
# The code is based on the modules 3 and 4. We use the same dataset: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# In[1]:


import pandas as pd
import numpy as np
import sklearn


# In[2]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[3]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[4]:


data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data_url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[5]:


y_train = df.churn


# In[6]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[7]:


dv = DictVectorizer()

train_dict = df[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


# In[8]:


customer = {
    'gender': 'male',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'yes',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 6,
    'monthlycharges': 29.85,
    'totalcharges': 129.85
}


# In[9]:


X = dv.transform(customer)


# In[10]:


model.predict_proba(X)[0, 1]


# In[23]:


import pickle


# In[12]:


with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# In[13]:


get_ipython().system('ls -lh')


# In[14]:


with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


# In[15]:


dv


# In[16]:


model


# In[17]:


customer = {
    'gender': 'male',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'yes',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 6,
    'monthlycharges': 29.85,
    'totalcharges': 129.85
}


# In[18]:


X = dv.transform(customer)


# In[19]:


churn = model.predict_proba(X)[0,1]


# In[20]:


churn


# In[21]:


if churn >= 0.5:
    print('send email with promo')
else:
    print('dont do anything')


# In[22]:


from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)

pipeline.fit(train_dict, y_train)


# In[24]:


with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)


# In[25]:


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


# In[28]:


churn = pipeline.predict_proba(customer)[0,1]


# In[29]:


churn


