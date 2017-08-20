# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:04:20 2017

@author: yliang
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
train_data  = pd.read_csv("E:/data/Kaggle_Benz/train.csv",sep = ',')

y_train = train_data['y']

x = train_data.drop(['ID','y'],axis=1)

for c in train_data.columns:
    if train_data[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train_data[c].values))
        train_data[c] = lbl.transform(list(train_data[c].values))

# shape        

x = train_data.drop(['ID','y'],axis=1)
y_mean = np.mean(y_train)

import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
dtrain = xgb.DMatrix(x, y_train)



# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=1000)
rf = RandomForestRegressor(max_depth=4,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=1000,)
rf.fit(x, y_train)



# check f2-score (to get higher score - increase num_boost_round in previous cell)


# now fixed, correct calculation
print(r2_score(y_train, model.predict(dtrain)))

print(r2_score(y_train, rf.predict(x)))
#print(r2_score(y_train, Ext_tree.predict(x)))

