import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def train_cbc(X_train , y_train) :

    cat_cols = X_train.select_dtypes(include = ["string" , "object" , "category"]).columns

    preprocess = ColumnTransformer([
        ("cat" , OneHotEncoder(handle_unknown = "ignore") , cat_cols)
    ] , remainder = "passthrough")

    model = Pipeline([
        ("preprocess" , preprocess) ,
        ("cbc" , CatBoostClassifier(
            iterations = None ,
            learning_rate = None ,
            depth = None ,
            l2_leaf_reg = None , 
            random_strength = None ,
            border_count = None ,
            boosting_type = "Plain" ,
            grow_policy = "Depthwise" ,
            eval_metric = "AUC" ,
            loss_function = "Logloss" ,
            random_state = 42 ,
            verbose = 0 ,
            task_type = "CPU"
        ))
    ])

    model.fit(
        X_train ,
        y_train
    )

    return model

