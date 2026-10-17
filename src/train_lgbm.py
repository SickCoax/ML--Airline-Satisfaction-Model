from lightgbm import LGBMClassifier

def get_lgbm(X_train , y_train) :
    
    cat_cols = X_train.select_dtypes(include = ["string" , "object" , "category"]).columns

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")

    model = LGBMClassifier(
        n_estimators = 2257 ,
        learning_rate = 0.01400207637043854 ,
        num_leaves = 81 ,
        max_depth = 5 ,
        min_child_samples = 194 ,
        min_child_weight = 0.00023547227584805327 ,
        subsample = 0.8078157370581776 ,
        subsample_freq = 4 ,
        colsample_bytree = 0.8429699233588869 ,
        colsample_bynode = 0.8422750811378343 ,
        extra_trees = False ,
        reg_alpha = 6.7879676933594535 ,
        reg_lambda = 5.356757716738945e-08 ,
        min_split_gain = 0.0466383349781442 ,
        path_smooth = 6.72941933379196 ,
        max_bin = 504 ,
        min_data_in_bin = 78 ,
        cat_smooth = 2.6421734086304727 ,
        cat_l2 = 0.0011519382614044997 ,
        max_cat_threshold = 66 ,
        max_cat_to_onehot = 3 ,
        objective = "binary" ,
        metric = "auc" ,
        boosting_type = "gbdt" ,
        importance_type = "gain" ,
        random_state = 42 ,
        verbosity = -1 ,
        n_jobs = -1
    )

    model.fit(  
        X_train ,
        y_train
    )

    return model