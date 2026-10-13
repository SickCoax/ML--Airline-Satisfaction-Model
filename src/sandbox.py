import os
import numpy as np
import pandas as pd
from preprocessing import get_X_y 
from train_xgbc import get_model
from sklearn.model_selection import cross_val_score , StratifiedKFold
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "train.csv"
)
df_train = pd.read_csv(csv_path)

X_train , y_train = get_X_y(df_train)

# print(X_train.columns)

cat_cols = X_train.select_dtypes(include = ["string" , "object"]).columns

# preprocess = ColumnTransformer([
#     ("cat" , OneHotEncoder(handle_unknown = "ignore") , cat_cols)
# ] , remainder = "passthrough")

# model = Pipeline([
#     ("preprocess" , preprocess) ,
#     ("xgbc" , XGBClassifier(
#         n_jobs = 1 ,
#         subsample = 0.8 ,
#         colsample_bytree = 0.8 ,
#         random_state = 42
#     ))
# ])

# cv = StratifiedKFold(
#     n_splits = 5 ,
#     shuffle = True ,
#     random_state = 42
# )

# scores = cross_val_score(
#     model ,
#     X_train ,
#     y_train ,
#     cv = cv ,
#     scoring = "roc_auc" ,
#     n_jobs = -1
# )

# print(scores)
# print(scores.mean())

cat_cols = X_train.select_dtypes(include = ["string" , "object" , "category"]).columns.to_list()

model = CatBoostClassifier(
    verbose = 0 ,
    random_state = 42 ,
    cat_features = cat_cols ,
    thread_count=1 
)

# cv = StratifiedKFold(
#     n_splits = 5 ,
#     shuffle = True ,
#     random_state = 42
# )

# scores = cross_val_score(
#     model ,
#     X_train ,
#     y_train ,
#     cv = cv ,
#     scoring = "roc_auc" ,
#     n_jobs = -1
# )

# print(scores)
# print(scores.mean())

model.fit(
    X_train ,
    y_train
)

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(
    X_train , 
    y_train , 
    test_size = 0.2 ,
    random_state = 42 ,
    stratify = y_train
)

y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print("ROC-AUC:", auc)