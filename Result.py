# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:13:25 2017

@author: MainPc
"""
#import library
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
import numpy as np
pd.options.mode.chained_assignment = None
import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

#datetime decomposition
datetime_variable = [lambda x : x.split()[0],  
     lambda x : x.split()[1].split(":")[0],
     lambda x : x.split()[0].split("-")[0],
     lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday(),
     lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month]

def feature_time_extract(data):
    data["date"] = data.datetime.apply(datetime_variable[0])
    data["hour"] = data.datetime.apply(datetime_variable[1]).astype("int")
    data["year"] = data.datetime.apply(datetime_variable[2])
    data["weekday"] = data.date.apply(datetime_variable[3])
    data["month"] = data.date.apply(datetime_variable[4])
    return data

#data type convertion
def load_features():
    train = pd.read_csv("c:/train.csv")
    test = pd.read_csv("c:/test.csv")
    feature_time_extract(train) 
    feature_time_extract(test) 
    for var in ["season","holiday","workingday","weather","weekday","month","year","hour"]:
        train[var] = train[var].astype("category")
        test[var] = test[var].astype("category")
    return train, test

#evaluation model
def rmsle(y, y_):
    return sqrt(mean_squared_error(np.log(np.exp(y) + 1),np.log(np.exp(y_) + 1)))

#selected features
feature_list = ["workingday","weather","weekday","month","hour","temp","humidity"]

#parameters for classifiers
ridge_parameters = {
    'max_iter':[1000],
    'alpha':[0.1, 1, 2, 3, 4, 10, 30, 100]
}

lasso_parameters = {
    'max_iter':[1000],
    'alpha':[0.1, 1, 2, 3, 4, 10, 30, 100]
}

xgb_parameters = {
    'n_estimators': [200, 150, 100]
}

rf_parameters = {
    'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
}

gbdt_parameters = {
    'n_estimators': [1000, 2000, 3000, 4000],
    'alpha': [0.01, 0.1, 0.2]
}


#linear regression
def lr (train):
    method = LinearRegression(train)
    logy = np.log1p(train["count"])
    method.fit(X=train, y=logy)
    prediction = method.predict(X = train)
    print("Linear Regression RMSLLE", rmsle(logy, prediction))
    
#ridge regression
def ridge_grid_search(train, test, parameter = ridge_parameters):
    method = Ridge()
    score = metrics.make_scorer(rmsle, greater_is_better=False)
    grid_search = GridSearchCV( method, parameter, scoring = score, cv=5)
    y = train["count"]
    log = np.log1p(y)
    grid_search.fit( train[feature_list], log )
    predeiction = grid_search.predict(X= train[feature_list])
    print (grid_search.best_params_)
    for i in grid_search.grid_scores_:
        print(i)
    print ("Ridge RMSLE",rmsle(log, predeiction))
    return predeiction

#lasso regression
def lasso_grid_search(train, test, parameter = lasso_parameters):
    method = Lasso()
    score = metrics.make_scorer(rmsle, greater_is_better=False)
    grid_search = GridSearchCV( method, parameter, scoring = score, cv=5)
    y = train["count"]
    log = np.log1p(y)
    grid_search.fit( train[feature_list], log )
    prediction = grid_search.predict(X= train[feature_list])
    print (grid_search.best_params_)
    for i in grid_search.grid_scores_:
        print(i)
    print ("Lasso RMSLE",rmsle(log, prediction))
    return prediction

#xgboost
def xgb_model(train,test):
    training = train[feature_list].iloc[:, :].values  
    offset = 8000  
    y = train["count"]
    log = np.log1p(y)
    train_RMSE = xgboost.DMatrix(training[:offset, :], label=log[:offset])  
    test_RMSE = xgboost.DMatrix(training[offset:, :], label=log[offset:])  
    resultlist = [(train_RMSE, 'Train'), (test_RMSE, 'Test')]  
    params = {"max_depth": 5, "tree_num": 1000, "silent": 1, "shrinkage": 0.01}  
    xgModel = xgboost.train(list(params.items()), train_RMSE, 200, resultlist, early_stopping_rounds=100)

#randome forest
def rf_grid_search(train, test, parameters = rf_parameters):
    method = RandomForestRegressor()
    score = metrics.make_scorer(rmsle, greater_is_better=False)
    grid_search = GridSearchCV(method, parameters, scoring = score, cv=5)
    y = train["count"]
    log = np.log1p(y)
    grid_search.fit( train[feature_list], log )
    prediction = grid_search.predict(X= train[feature_list])
    print (grid_search.best_params_)
    for i in grid_search.grid_scores_:
        print(i)
    print ("Random Forest RMSLE",rmsle(log, prediction))
    return prediction

#GBDT
def gbdt_grid_search(train, test, parameters = gbdt_parameters):
    method = GradientBoostingRegressor()
    score = metrics.make_scorer(rmsle, greater_is_better=False)
    grid_search = GridSearchCV(method,  parameters, scoring = score, cv=5)
    y = train["count"]
    log = np.log1p(y)
    grid_search.fit(train[feature_list], log )
    prediction = grid_search.predict(X=train[feature_list])
    print (grid_search.best_params_)
    for i in grid_search.grid_scores_:
        print(i)
    print ("GBDT RMSLE",rmsle(log, prediction))
    return prediction

#main function
if __name__ == "__main__":
    train, test = load_features()
    ridge_grid_search(train, test)
    lasso_grid_search(train, test)
    xgb_model(train,test)
    rf_grid_search(train, test)
    gbdt_grid_search(train, test)
    print()


