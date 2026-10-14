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
            iterations = 1971 ,
            learning_rate = 0.008513323279531235 ,
            depth = 10 ,
            l2_leaf_reg = 12.578821933374982 , 
            random_strength = 9.049273412474395 ,
            border_count = 251 ,
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