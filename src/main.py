import os
import pandas as pd
from preprocessing import get_X_y , get_X
from train_xgbc import get_xgbc
from train_cb import get_cbc


csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "train.csv"
)
df_train = pd.read_csv(csv_path)

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "test.csv"
)
df_test = pd.read_csv(csv_path)

X_train , y_train = get_X_y(df_train)
X_test = get_X(df_test)

xgbc_model = get_xgbc(
    X_train ,
    y_train
)

cbc_model = get_cbc(
    X_train ,
    y_train
)

id = df_test["id"].values
y_prob = xgbc_model.predict_proba(X_test)[ : , 1]


result = pd.DataFrame(
    id ,
    columns = ["id"]
)

result["satisfaction"] = y_prob

result.to_csv("result1.csv" , index=False)