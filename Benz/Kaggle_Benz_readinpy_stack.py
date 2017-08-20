# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:23:30 2017

@author: yliang
"""


import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

def add_new_col(x):
    if x not in new_col.keys(): 
        # set n/2 x if is contained in test, but not in train 
        # (n is the number of unique labels in train)
        # or an alternative could be -100 (something out of range [0; n-1]
        return int(len(new_col.keys())/2)
    return new_col[x] # rank of the label
    
train = pd.read_csv('E:/data/Kaggle_Benz/train.csv')
test = pd.read_csv('E:/data/Kaggle_Benz/test.csv')

n_tr = len(train)
cat_cols = []
for c in train.columns:
    if train[c].dtype == 'object':
        cat_cols.append(c)
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



onehot = pd.get_dummies(train['X0'])

for c in cat_cols:
    # get labels and corresponding means
    new_col = train.groupby(c).y.mean().sort_values().reset_index()
    # make a dictionary, where key is a label and value is the rank of that label
    new_col = new_col.reset_index().set_index(c).drop('y', axis=1)['index'].to_dict()
    # add new column to the dataframe
    #df[c + '_new'] = df[c].apply(add_new_col)
    train[c + '_new'] = train[c].apply(add_new_col)
    test[c + '_new'] = test[c].apply(add_new_col)


train = train.drop(cat_cols, axis=1)
test = test.drop(cat_cols, axis=1)

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

#usable_columns = list(set(train.columns) - set(['y']))



y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

'''
gaus = GaussianMixture(n_components=3, random_state=0)
gaus.fit(y_train)
labels = gaus.predict((tsne_representation[:n_tr]))
print('Gaussian Mixture:')
show_plots(labels)                   
   '''                

'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 600, 
    'eta': 0.0045,
    'max_depth': 5,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
# NOTE: Make sure that the class is labeled 'class' in the data file
train_2 = np.hstack((train.drop('y', axis=1), onehot))
dtrain = xgb.DMatrix(train_2, y_train)
dtrain2 = xgb.DMatrix(finaltrainset, y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
model1 = xgb.train(dict(xgb_params, silent=0), dtrain2, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)
y_pred1 = model.predict(dtest)

'''Train the stacked models then predict the test data'''
import lightgbm as lgb  

train_data = lgb.Dataset(train.drop('y', axis=1), y_train, feature_name=list(train.drop('y', axis=1).columns), categorical_feature=cat_cols)
param = {'max_depth':6,'num_leaves':64,'learning_rate':0.03,'scale_pos_weight':1,'num_threads':40,'objective':'binary','bagging_fraction':0.7,'bagging_freq':1,'min_sum_hessian_in_leaf':100}
estimators = lgb.train(param,train_data,num_boost_round=100)




stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=RandomForestRegressor(max_depth=4,n_estimators=100)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)
#use lasso to combine the result
GBM2 = GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.9)
GBM2.fit(finaltrainset, y_train)


stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.3 + model.predict(dtrain)*0.7))
print(r2_score(y_train, model.predict(dtrain)))
print(r2_score(y_train, model1.predict(dtrain2)))
print(r2_score(y_train, model1.predict(dtrain2)*0.25+model.predict(dtrain)*0.75))

print(r2_score(y_train, estimators.predict(train.drop('y', axis=1))))
print(r2_score(y_train, stacked_pipeline.predict(finaltrainset)))
print(r2_score(y_train, GBM2.predict(finaltrainset)))

'''Average the preditionon test data  of both models then save it on a csv file'''
y_pred1 = model1.predict(dtrain2)
y_pred2 = model.predict(dtrain)
sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)




