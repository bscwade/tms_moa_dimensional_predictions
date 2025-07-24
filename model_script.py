#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import itertools as it
import argparse

# get command line args
parser=argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=True)
args=parser.parse_args()
index=args.index

# set parameters
outcome_list=['hamd17_global', 'hdrs6', 'cma', 'sod', 'ins']
outcome=outcome_list[index]
folds=10

# neuropsych data
neuropsych=pd.read_csv('/path/to/data.csv')

# add hdrs subscale outcomes
neuropsych['hdrs6']=neuropsych.loc[:, ['hamd28_1', 'hamd28_2', 'hamd28_7', 'hamd28_8', 'hamd28_10', 'hamd28_13']].sum(axis=1, skipna=False)
neuropsych['cma']=neuropsych.loc[:, ['hamd28_1', 'hamd28_7', 'hamd28_8', 'hamd28_16']].sum(axis=1, skipna=False)
neuropsych['sod']=neuropsych.loc[:, ['hamd28_2', 'hamd28_9', 'hamd28_10', 'hamd28_11', 'hamd28_12','hamd28_13', 'hamd28_14', 'hamd28_15']].sum(axis=1, skipna=False)
neuropsych['ins']=neuropsych.loc[:, ['hamd28_4', 'hamd28_5', 'hamd28_6']].sum(axis=1, skipna=False)

neuropsych.loc[:, 'participant_id'] = [x.replace('_', '') for x in neuropsych.loc[:, 'participant_id']]

# rsfc data
rsfc=pd.read_csv('/ath/to/rsfc.csv', index_col=[0])


# format data
timepoint_bool = (neuropsych.loc[:, 'redcap_event_name'] == 'visit_1_arm_3') | (neuropsych.loc[:, 'redcap_event_name'] == 'visit_3_arm_3')
neuropsych=neuropsych.loc[timepoint_bool, :]
outcome_bool=neuropsych.loc[:,outcome].isna()
neuropsych=neuropsych[outcome_bool==False]
id_keep=pd.Series(neuropsych['participant_id'].value_counts()==2).index[neuropsych['participant_id'].value_counts()==2]
neuropsych=neuropsych[[x in id_keep for x in neuropsych.loc[:, 'participant_id']]]


np_df=pd.DataFrame({'participant_id': neuropsych[neuropsych['redcap_event_name'] == 'visit_3_arm_3']['participant_id'],
                    'outcome_change': np.array(neuropsych[neuropsych['redcap_event_name'] == 'visit_3_arm_3'][outcome]) - np.array(neuropsych[neuropsych['redcap_event_name'] == 'visit_1_arm_3'][outcome])})


# merge rsfc
clf_data=np_df.merge(rsfc, on='participant_id')
clf_data=clf_data.drop(['participant_id'], axis=1)

# set X, y
y=clf_data.pop('outcome_change')
X=clf_data

# Nested CV with parameter optimization
inner_cv = KFold(n_splits=folds, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=folds, shuffle=True, random_state=1)


# set pipeline and parameter grid    
estimators=[('imputer', SimpleImputer(strategy='median')), 
        ('selector', SelectKBest(mutual_info_regression)), 
        ('classifier', RandomForestRegressor())]

params={
    "selector__k": [10, 50, 100, X.shape[1]],
    "classifier__n_estimators": [50, 100, 500, 1000],
    "classifier__max_features": [1, 5, 10, 1.0, 10.0, 'sqrt', 'log2', 'none']        
}
    

pipe=Pipeline(estimators)
clf=GridSearchCV(estimator=pipe, param_grid=params, cv=inner_cv)

# fit model
preds=cross_val_predict(clf, X, y.ravel(), cv=outer_cv, n_jobs=1, verbose=1) 
perf=r2_score(y, preds)   



