#This script was redacted by Ramzi Bouyekhf (propower)

# Train the model base 1 with xgboost, then select top 380 best features and save new train/test dataframes.

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc


def features_selection():
    gc.enable()
    
    # load handcrafted dataframes (train/test)
    train = pd.read_csv("processed/train_model_base_1.csv").set_index("SK_ID_CURR")
    test = pd.read_csv("processed/test_model_base_1.csv").set_index("SK_ID_CURR")
    
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # feats is the list of features that will be used for training
    feats_xgb = [f for f in train.columns if f not in ['SK_ID_CURR','TARGET']]
    
    # vector that will store the importance of each feature. Used to analyse the most important feature in our final model
    feats_importance_xgb = np.zeros(len(feats_xgb))

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
        trn_x, trn_y = train[feats_xgb].iloc[trn_idx], train.iloc[trn_idx]['TARGET']
        val_x, val_y = train[feats_xgb].iloc[val_idx], train.iloc[val_idx]['TARGET']

        # The parameters were taken from public kernels then tuned manually to increase accuracy
        clf = XGBClassifier(
            n_estimators=30000,
            learning_rate = 0.02,
            num_leaves = 35,
            colsample_bytree = 0.1,
            subsample = 0.9,
            max_depth = 7,
            reg_alpha = 0.041545473,
            reg_lambda = 0.0735294,
            min_split_gain = 0.022,
            min_child_weight = 32,
            n_jobs=24
        )
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
               )

        feats_importance_xgb += clf.feature_importances_ / folds.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds_xgb[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    # create features importance dataframe then select top 380 features
    feature_importance = pd.DataFrame({"column_name" : feats_xgb, "feature_importance" : feats_importance_xgb})
    feature_importance = feature_importance.sort_values("feature_importance", ascending = False)
    cols = feature_importance.head(380)["column_name"].tolist()
    
    # select subset of features using top 380 features for train and test
    test = test[cols]
    test.to_csv("processed/test_model_base_1.csv")
    
    # same for train, but add target column
    cols.append("TARGET")
    train = train[cols]
    train.to_csv("processed/train_model_base_1.csv")
