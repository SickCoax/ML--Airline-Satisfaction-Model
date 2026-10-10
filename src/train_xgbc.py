import os
from preprocessing import get_X_y
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def get_model(X_train , y_train) : 

    cat_cols = X_train.select_dtypes(include = ["string" , "object"]).columns

    preprocess = ColumnTransformer([
        ("cat" , OneHotEncoder(handle_unknown = "ignore") , cat_cols)
    ] , remainder = "passthrough")

    model = Pipeline([
        ("preprocess" , preprocess) ,
        ("xgbc" , XGBClassifier(
            n_jobs = -1 ,
            subsample = 0.9260074439454506 ,
            colsample_bytree = 0.7104877345208437 ,
            random_state = 42 ,
            n_estimators = 1423 ,
            max_depth = 11 ,
            learning_rate = 0.013735365034512911 ,
            min_child_weight = 13 ,
            gamma = 1.9999543167310362e-06 ,
            reg_alpha = 6.906442415150357 ,
            reg_lambda = 9.5312610099552e-05
        ))
    ])

    model.fit(
        X_train ,
        y_train
    )

    return model