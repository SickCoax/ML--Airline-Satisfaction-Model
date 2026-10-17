from lightgbm import LGBMClassifier

def get_lgbm(X_train , y_train) :
    
    cat_cols = X_train.select_dtypes(include = ["string" , "object" , "category"]).columns

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")

        model = LGBMClassifier()

        model.fit(
            X_train ,
            y_train
        )

        return model