# This script was redacted by Ramzi Bouyekhf (propower)

import pandas as pd 
import numpy as np
import random
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc

def lightgbm(train, test, seed, model_name):
    gc.enable()
    print(train.shape)
    
    # create folds for cross validation and vectors for out-of-folds predictions and for submissions
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds_lgbm = np.zeros(train.shape[0])
    sub_preds_lgbm = np.zeros(test.shape[0])
    
    # feats is the list of features that will be used for training
    feats = [f for f in train.columns if f not in ['SK_ID_CURR','TARGET']]
    
    # vector that will store the importance of each feature. Used to analyse the most important feature in our final model
    feats_importance_lgbm = np.zeros(len(feats))
    train_preds = np.zeros(train.shape[0])

    # 5 folds for cross validation
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
        trn_x, trn_y = train[feats].iloc[trn_idx], train.iloc[trn_idx]['TARGET']
        val_x, val_y = train[feats].iloc[val_idx], train.iloc[val_idx]['TARGET']
        
        # The parameters were taken from public kernels then tuned manually
        clf = LGBMClassifier(
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
            n_jobs=-1
        )

        # Function to train and validate on each fold created
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=100, early_stopping_rounds=300  #30
               )

        # fill out-of-folds predictions
        oof_preds_lgbm[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        
        # fill submission predictions. The final submission is the average of the prediction of 5 models trained with cross validation
        sub_preds_lgbm += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        
        # fill importance features vector
        feats_importance_lgbm += clf.feature_importances_ / folds.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds_lgbm[val_idx])))
        
        # remove variables that won't be useful anymore to decrease memory usage
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    # Put CV out-of-folds predictions and submissions predictions in dataframes and save results in csv
    CV = pd.DataFrame({"SK_ID_CURR" : train.index, "lgbm_" + model_name + "_" + str(seed) : oof_preds_lgbm})
    CV.to_csv("processed/cv_lgbm_" + model_name + "_" + str(seed) + ".csv", index = False)
    test_results = pd.DataFrame({ 'SK_ID_CURR': test.index,"lgbm_" + model_name + "_" + str(seed): sub_preds_lgbm })
    test_results.to_csv("processed/test_lgbm_" + model_name + "_" + str(seed) + ".csv", index = False)
    print('Final auc ' + str(roc_auc_score(train["TARGET"], oof_preds_lgbm)))
    
# same procedure in xgboost and lightgbm in training
def xgboost(train, test, seed, model_name):
    gc.enable()

    for column in train.columns:
        if train[column].dtype == "object":
            train[column] = train[column].astype('category')

    for column in test.columns:
        if test[column].dtype == "object":
            test[column] = test[column].astype('category')

    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    test["CODE_GENDER_XNA"] = 0
    test["NAME_INCOME_TYPE_Maternity leave"] = 0
    test["NAME_FAMILY_STATUS_Unknown"] = 0

    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_xgb = np.zeros(train.shape[0])
    sub_preds_xgb = np.zeros(test.shape[0])
    feats_xgb = [f for f in train.columns if f not in ['SK_ID_CURR','TARGET']]
    feats_importance_xgb = np.zeros(len(feats_xgb))
    train_preds = np.zeros(train.shape[0])

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
        trn_x, trn_y = train[feats_xgb].iloc[trn_idx], train.iloc[trn_idx]['TARGET']
        val_x, val_y = train[feats_xgb].iloc[val_idx], train.iloc[val_idx]['TARGET']

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

        oof_preds_xgb[val_idx] = clf.predict_proba(val_x)[:, 1]
        sub_preds_xgb += clf.predict_proba(test[feats_xgb])[:, 1] / folds.n_splits
        feats_importance_xgb += clf.feature_importances_ / folds.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds_xgb[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    CV = pd.DataFrame({"SK_ID_CURR" : train.index, "xgb_" + model_name + "_" + str(seed) : oof_preds_xgb})
    CV.to_csv("processed/cv_xgb_" + model_name + "_" + str(seed) + ".csv", index = False)
    test_results = pd.DataFrame({ 'SK_ID_CURR': test.index,"xgb_" + model_name + "_" + str(seed): sub_preds_xgb })
    test_results.to_csv("processed/test_xgb_" + model_name + "_" + str(seed) + ".csv", index = False)
    print('Final auc ' + str(roc_auc_score(train["TARGET"], oof_preds_xgb)))
    
    
def get_train_test(train,test, train_gp_file, test_gp_file):
    gpfeats = pd.read_csv(train_gp_file)
    gpfeats.index = train.index
    train = train.join(gpfeats)

    gpfeats = pd.read_csv(test_gp_file)
    gpfeats.index = test.index
    test = test.join(gpfeats)
    return train, test


def prepare_final_models():
    # train with model base 1. Add a set of features created with genetic programming to model base 1 and train with lightgbm and xgboost in each time 
    model_name = "model_1"
    df_train = pd.read_csv("processed/train_model_base_1.csv")
    df_test = pd.read_csv("processed/test_model_base_1.csv")
    df_train = df_train.set_index("SK_ID_CURR")
    df_test = df_test.set_index("SK_ID_CURR")
  
    #use gp features with model base 1 
    train = df_train.copy()
    test = df_test.copy()
    train, test = get_train_test(train,test, "processed/train_gp_features.csv", "processed/test_gp_features.csv")
#     seed = 49
#     lightgbm(train, test, seed, model_name)
    seed = 169
    xgboost(train, test, seed, model_name)
    
    #use gp1 features with model base 1
    train = df_train.copy()
    test = df_test.copy()
    train, test = get_train_test(train,test, "processed/train_gp1_features.csv", "processed/test_gp1_features.csv")
    seed = 45
    lightgbm(train, test, seed, model_name)
#     seed = 52
#     xgboost(train, test, seed, model_name)
    
    #use gp2 features with model base 1 
    train = df_train.copy()
    test = df_test.copy()
    train, test = get_train_test(train,test, "processed/train_gp2_features.csv", "processed/test_gp2_features.csv")
    seed = 33
    lightgbm(train, test, seed, model_name)
    seed = 20
    xgboost(train, test, seed, model_name)
    
    #use gp3 features with model base 1 
    train = df_train.copy()
    test = df_test.copy()
    train, test = get_train_test(train,test, "processed/train_gp3_features.csv", "processed/test_gp3_features.csv")
    seed = 1001
    lightgbm(train, test, seed, model_name)
    seed = 512
    xgboost(train, test, seed, model_name)
    
    # train with model base 2. Add a set of features created with genetic programming to model base 1 and train with lightgbm and xgboost in each time 
    model_name = "model_2"
    df_train = pd.read_csv("processed/train_model_base_2.csv")
    df_test = pd.read_csv("processed/test_model_base_2.csv")
    df_train = df_train.set_index("SK_ID_CURR")
    df_test = df_test.set_index("SK_ID_CURR")
    
    #use gp features with model base 2 
#     train = df_train.copy()
#     test = df_test.copy()
#     train, test = get_train_test(train,test, "processed/train_gp_features.csv", "processed/test_gp_features.csv")
#     seed = 754
#     lightgbm(train, test, seed, model_name)
#     seed = 189
#     xgboost(train, test, seed, model_name)
    
    #use gp1 features with model base 2
    train = df_train.copy()
    test = df_test.copy()
    train, test = get_train_test(train,test, "processed/train_gp1_features.csv", "processed/test_gp1_features.csv")
    seed = 42
    lightgbm(train, test, seed, model_name)
    seed = 86
    xgboost(train, test, seed, model_name)
    
    #use gp2 features with model base 2
    train = df_train.copy()
    test = df_test.copy()
    train, test = get_train_test(train,test, "processed/train_gp2_features.csv", "processed/test_gp2_features.csv")
    seed = 83
    lightgbm(train, test, seed, model_name)
#     seed = 615
#     xgboost(train, test, seed, model_name)
    
    #use gp3 features with model base 2
    train = df_train.copy()
    test = df_test.copy()
    train, test = get_train_test(train,test, "processed/train_gp3_features.csv", "processed/test_gp3_features.csv")
    seed = 69
    lightgbm(train, test, seed, model_name)
    seed = 437
    xgboost(train, test, seed, model_name)
    
