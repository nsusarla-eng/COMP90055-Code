# -*- coding: utf-8 -*-

#Created on Tue May 26 17:27:51 2020

#For census dataset, one-hot encoding of all attributes apart from age and hours-per-week
#and scaling of age and hours-per-week attributes.  

#Reference: https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/ch04.html

#@author: nandita


import os
import mglearn
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"

adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")

data = pd.read_csv(adult_path, header=None, index_col=False,names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'],na_values=' ?',keep_default_na=False)

data = data.dropna()

# Reducing categories in education feature
data = data.replace(dict.fromkeys(['1st-4th','5th-6th'], 'Primary'),regex=True)
data = data.replace(dict.fromkeys(['7th-8th','9th','10th','11th','12th'], 'High'),regex=True)
data = data.replace(dict.fromkeys(['HS-grad','Some-college','Assoc-voc','Assoc-acdm','Prof-school'], 'Vocation'),regex=True)

# Reducing categories in occupation feature
data = data.replace(dict.fromkeys(['Tech-support'], 'Prof-specialty'),regex=True)
data = data.replace(dict.fromkeys(['Craft-repair','Machine-op-inspct','Transport-moving'], 'Mechanical'),regex=True)
data = data.replace(dict.fromkeys(['Priv-house-serv'], 'Handlers-cleaners'),regex=True)
data = data.replace(dict.fromkeys(['Farming-fishing','Armed-Forces'], 'Other-service'),regex=True)

# Reducing categories in workclass feature
data = data.replace(dict.fromkeys(['Self-emp-not-inc','Self-emp-inc'], 'Self-emp'),regex=True)
data = data.replace(dict.fromkeys(['Local-gov','State-gov','Federal-gov'], 'Government'),regex=True)
data = data.replace(dict.fromkeys(['Without-pay'], 'Voluntary'),regex=True)

data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation','race','income']]

ct = ColumnTransformer(
    [("scaling", StandardScaler(), ['age','hours-per-week']),
     ("onehot", OneHotEncoder(sparse=False),
     ['education','gender','workclass','occupation','race','income'])])
  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ct.fit(data)
data_feat_trans = ct.transform(data)
print(data_feat_trans.shape)
data_labels = data_feat_trans[:,28]
data_feats = data_feat_trans[:,:28]
print(data_feats.shape)
print(data_labels.shape)


CENSUS_PATH = './censusdata/'

if not os.path.exists(CENSUS_PATH):
    os.makedirs(CENSUS_PATH)
    
np.savetxt(CENSUS_PATH + 'Censusfeatsnew.csv',data_feats, delimiter=',')
np.savetxt(CENSUS_PATH + 'Censuslabelsnew.csv',data_labels, delimiter=',')


