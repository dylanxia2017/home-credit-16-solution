# This script was redacted by Ramzi Bouyekhf (propower)

import pandas as pd 
import numpy as np

def stacking():
    tr = pd.read_csv("input/application_train.csv").set_index("SK_ID_CURR")
#     cv1 = pd.read_csv("cv_lgbm_model_1_49.csv").set_index("SK_ID_CURR")
    cv2 = pd.read_csv("processed/cv_xgb_model_1_169.csv").set_index("SK_ID_CURR")
    cv3 = pd.read_csv("processed/cv_lgbm_model_1_45.csv").set_index("SK_ID_CURR")
#     cv4 = pd.read_csv("cv_xgb_model_1_52.csv").set_index("SK_ID_CURR")
    cv5 = pd.read_csv("processed/cv_lgbm_model_1_33.csv").set_index("SK_ID_CURR")
    cv6 = pd.read_csv("processed/cv_xgb_model_1_20.csv").set_index("SK_ID_CURR")
    cv7 = pd.read_csv("processed/cv_lgbm_model_1_1001.csv").set_index("SK_ID_CURR")
    cv8 = pd.read_csv("processed/cv_xgb_model_1_512.csv").set_index("SK_ID_CURR")
#     cv9 = pd.read_csv("cv_lgbm_model_2_754.csv").set_index("SK_ID_CURR")
#     cv10 = pd.read_csv("cv_xgb_model_2_189.csv").set_index("SK_ID_CURR")
    cv11 = pd.read_csv("processed/cv_lgbm_model_2_42.csv").set_index("SK_ID_CURR")
    cv12 = pd.read_csv("processed/cv_xgb_model_2_86.csv").set_index("SK_ID_CURR")
    cv13 = pd.read_csv("processed/cv_lgbm_model_2_83.csv").set_index("SK_ID_CURR")
#     cv14 = pd.read_csv("cv_xgb_model_2_615.csv").set_index("SK_ID_CURR")
    cv15 = pd.read_csv("processed/cv_lgbm_model_2_69.csv").set_index("SK_ID_CURR")
    cv16 = pd.read_csv("processed/cv_xgb_model_2_437.csv").set_index("SK_ID_CURR")
    train = tr[["TARGET"]].join(cv2)\
        .join(cv3)\
        .join(cv5)\
        .join(cv6)\
        .join(cv7)\
        .join(cv8)\
        .join(cv11)\
        .join(cv12)\
        .join(cv13)\
        .join(cv15)\
        .join(cv16)
    
#     te1 = pd.read_csv("test_lgbm_model_1_49.csv").set_index("SK_ID_CURR")
    te2 = pd.read_csv("processed/test_xgb_model_1_169.csv").set_index("SK_ID_CURR")
    te3 = pd.read_csv("processed/test_lgbm_model_1_45.csv").set_index("SK_ID_CURR")
#     te4 = pd.read_csv("test_xgb_model_1_52.csv").set_index("SK_ID_CURR")
    te5 = pd.read_csv("processed/test_lgbm_model_1_33.csv").set_index("SK_ID_CURR")
    te6 = pd.read_csv("processed/test_xgb_model_1_20.csv").set_index("SK_ID_CURR")
    te7 = pd.read_csv("processed/test_lgbm_model_1_1001.csv").set_index("SK_ID_CURR")
    te8 = pd.read_csv("processed/test_xgb_model_1_512.csv").set_index("SK_ID_CURR")
#     te9 = pd.read_csv("test_lgbm_model_2_754.csv").set_index("SK_ID_CURR")
#     te10 = pd.read_csv("test_xgb_model_2_189.csv").set_index("SK_ID_CURR")
    te11 = pd.read_csv("processed/test_lgbm_model_2_42.csv").set_index("SK_ID_CURR")
    te12 = pd.read_csv("processed/test_xgb_model_2_86.csv").set_index("SK_ID_CURR")
    te13 = pd.read_csv("processed/test_lgbm_model_2_83.csv").set_index("SK_ID_CURR")
#     te14 = pd.read_csv("test_xgb_model_2_615.csv").set_index("SK_ID_CURR")
    te15 = pd.read_csv("processed/test_lgbm_model_2_69.csv").set_index("SK_ID_CURR")
    te16 = pd.read_csv("processed/test_xgb_model_2_437.csv").set_index("SK_ID_CURR")

    test = te2.join(te3)\
        .join(te5)\
        .join(te6)\
        .join(te7)\
        .join(te8)\
        .join(te11)\
        .join(te12)\
        .join(te13)\
        .join(te15)\
        .join(te16)
        
    print(train.shape)
    print(test.shape)
    
    from sklearn.linear_model import LogisticRegression

    cols = [col for col in train.columns if(col != "TARGET")]
    x_train = train[cols]
    y_train = train["TARGET"]
    x_test = test[cols]

    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train,y_train)

    x_test['TARGET'] = logistic_regression.predict_proba(x_test)[:,1]
    x_test = x_test.reset_index()

    x_test[['SK_ID_CURR', 'TARGET']].to_csv('processed/final_submission.csv', index=False)