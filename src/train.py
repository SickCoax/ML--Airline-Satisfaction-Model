import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X_train = None
y_train = None

cat_cols = X_train.select_dtypes(include = ["string" , "object"]).columns

preprocess = ColumnTransformer([
    ("cat" , OneHotEncoder(handle_unknown = "ignore") , cat_cols)
] , remainder = "passthrough")

model = Pipeline([
    ("preprocess" , preprocess) ,
    ("xgbc" , XGBClassifier(
        n_jobs = -1 ,
        subsample = 0.8 ,
        colsample_bytree = 0.8 ,
        random_state = 42
    ))
])

model.fit(
    X_train ,
    y_train
)
