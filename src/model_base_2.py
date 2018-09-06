# This script was redacted by Toaru - https://www.kaggle.com/marrvolo

import gc
import pandas as pd
import numpy as np
import pickle

def generate_agg_feature(dataset, aggregation, prefix, groupby="SK_ID_CURR"):
    dataset_agg = dataset.groupby(groupby).agg(aggregation)
    dataset_agg.columns = pd.Index([prefix + e[0] + "_" + e[1].upper() for e in dataset_agg.columns.tolist()])
    return dataset_agg

def filter_weak_feature(df, level):
    weak_list = pickle_load(df, "utils/weak_feature_%d.pkl" % level)
    drop_list = []
    for n, feat in enumerate(weak_list):
        if feat in df.columns:
            drop_list.append(feat)
    df.drop(drop_list, axis=1, inplace=True)
    return df
    
def one_hot_encoder(df, nan_as_category = True):
    cate_filter_list = [
    "CREDIT_CURRENCY"
    ]
    df = df.drop([col for col in df.columns if col in cate_filter_list], axis=1)
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def load_and_clean_application_train_test(num_rows = None, nan_as_category = True):
    df = pd.read_csv('input/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index(drop=True)
    
    print("Dataset clean")
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    print("Cate feature specify")
    cate_feature = []
    cate_feature.extend([column for column in df.columns
                 if "FLAG" in column
                 or "NAME" in column
                 or "_NOT_" in column
                 or "_TYPE" in column])
    cate_feature.extend(["CODE_GENDER", "FONDKAPREMONT_MODE", 
                 "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"])

    for col in ["FONDKAPREMONT_MODE", "NAME_EDUCATION_TYPE", "NAME_TYPE_SUITE",
                "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_INCOME_TYPE", 
                "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "WALLSMATERIAL_MODE"]:
        cate_feature.remove(col)
    # Categorical features: Binary features and One-Hot encoding
    for bin_feature in cate_feature:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])    
        #df[bin_feature] = df[bin_feature].astype("category")
        
    education_encoder = {
        "Secondary / secondary special": 1,
        "Academic degree": 4,
        "Higher education": 3,
        "Incomplete higher": 2,
        "Lower secondary": 0,
    }

    week_encoder = {
        "FRIDAY": 5,
        "MONDAY": 1,
        "SATURDAY": 6,
        "SUNDAY": 7,
        "THURSDAY": 4,
        "TUESDAY": 2,
        "WEDNESDAY": 3,
    }
    
    df["WEEKDAY_APPR_PROCESS_START"] = df["WEEKDAY_APPR_PROCESS_START"].apply(lambda x: week_encoder.get(x, -1))
    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].apply(lambda x: education_encoder.get(x, -1))
    # df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    df, _ = one_hot_encoder(df)
    
    df.index = df.SK_ID_CURR
    
    del test_df
    gc.collect()
    return df, cate_feature

def application_feature_generation(df, cate_feature):
    df = df.copy()
    
    print("Sample transform")
    #df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    #df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    #df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    #df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    columns = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE", 
     "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", 
     "EXT_SOURCE_3"]
    
    for cola in columns:
        for colb in columns:
            if cola != colb:
                df["%s_DIV_%s" % (cola, colb)] = df[cola] / df[colb]

    return df

def load_and_clean_bureau_and_balance(num_rows = None, nan_as_category = True):
    print("Load dataset")
    bureau = pd.read_csv('input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('input/bureau_balance.csv', nrows = num_rows)
    bb["NUM_STATUS"] = bb["STATUS"].replace(["C", "X"], "0").astype(int)
    
    bureau_curr_mapping = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].drop_duplicates()
    
    print("Mini nettoyage")
    loan_type = bureau.groupby("CREDIT_TYPE").SK_ID_BUREAU.count()
    loan_type = loan_type[loan_type < 100].index
    bureau.loc[bureau.CREDIT_TYPE.isin(loan_type), "CREDIT_TYPE"] = "Unknown type of loan"
    bureau.loc[bureau.CREDIT_CURRENCY != "currency 1", "CREDIT_CURRENCY"] = "others"
    bureau.loc[bureau.CREDIT_ACTIVE == "Bad debt", "CREDIT_ACTIVE"] = None
    
    print("One hot encoder")
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    print("Agg bureau balance with bureau")
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    #bureau.drop(columns= 'SK_ID_BUREAU', inplace= True)
    del bb_agg
    gc.collect()

    bureau["DPD"] = bureau.DAYS_ENDDATE_FACT - bureau.DAYS_CREDIT_ENDDATE
    bureau["DBD"] = bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_ENDDATE_FACT
    bureau["DPD"] = bureau["DPD"].apply(lambda x: x if x > 0 else 0)
    bureau["DBD"] = bureau["DBD"].apply(lambda x: x if x > 0 else 0)
    
    bureau["DURING"] = bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_CREDIT
    bureau["DURING_FACT"] = bureau.DAYS_ENDDATE_FACT - bureau.DAYS_CREDIT
    bureau["DURING_PERC"] = bureau["DURING_FACT"] / bureau["DURING"]
    
    bureau["CREDIT_PERC"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"]
    bureau["CREDIT_DURING"] = bureau["AMT_CREDIT_SUM"] / bureau["DURING"]
    
    bb = bb.merge(bureau_curr_mapping, on="SK_ID_BUREAU")
    
    return bureau, bb, bureau_cat, bb_cat

def recent_bureau_feature_generation():
    print("Load dataset")
    bureau_origin = pd.read_csv('input/bureau.csv')
    bb_origin = pd.read_csv('input/bureau_balance.csv')
    bb_origin = bb_origin.merge(bureau_origin[["SK_ID_CURR", "SK_ID_BUREAU"]], on="SK_ID_BUREAU", how="right")
    res = pd.DataFrame(index=bureau_origin.SK_ID_CURR.unique())
    
    bb_aggregations = {
    
    }
    for status in ['0', '1', '2', '3', '4', '5', 'C', 'X']:
        bb_aggregations["STATUS_" + status] = ["mean"]
        
    aggregations = {
        "AMT_ANNUITY": ["sum"],
        "CREDIT_DAY_OVERDUE": ["mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "CNT_CREDIT_PROLONG": ["mean", "max"],
        "AMT_CREDIT_SUM": ["sum"],
        "AMT_CREDIT_SUM_DEBT": ["sum"],
        "AMT_CREDIT_SUM_LIMIT": ["sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"]
    }

    print("Generation")
    for time in [[0, 6], [6, 12], [12, 24]]:
        bb = bb_origin[(bb_origin.MONTHS_BALANCE <= -time[0]) & (bb_origin.MONTHS_BALANCE > -time[1])].copy()
        bb, _ = one_hot_encoder(bb, nan_as_category=False)
        
        bb_agg = generate_agg_feature(bb, bb_aggregations, 'BB_%d_' % time[0])
        res = res.join(bb_agg, how="left")

    bb = bb_origin[(bb_origin.MONTHS_BALANCE > -24)].copy()
    bureau = bureau_origin[bureau_origin.SK_ID_BUREAU.isin(bb.SK_ID_BUREAU.unique())].copy()
    bureau_agg = generate_agg_feature(bureau, aggregations, 'BURO_24_')
    res = res.join(bureau_agg, how="left")
    
    bb_origin = bb_origin[bb_origin.STATUS.isin(["1", "2", "3", "4", "5"])].copy()
    aggregations = {
        "MONTHS_BALANCE": ["max", "count"],
    }
    bb_origin_agg = generate_agg_feature(bb_origin, aggregations, 'BB_')
    res = res.join(bb_origin_agg, how="left")
    
    return res

def bureau_feature_generation(bureau, bureau_cat, bb_cat):
    bureau = bureau.copy()
    
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        "DPD": ["mean", "sum"],
        "DBD": ["mean", "max", "sum"],
        "DURING": ["min", "max", "mean"],
        "DURING_FACT": ["min", "max", "mean"],
        "DURING_PERC": ["min", "max", "mean"],
        "CREDIT_PERC": ["min", "max", "mean"],
        "CREDIT_DURING": ["min", "max", "mean"],
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: 
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat: 
        cat_aggregations[cat + "_MEAN"] = ['mean']
    
    print("Bureau cate feature / num feature aggregation")
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    

    print("Bureau active credit num feature aggregation")
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = generate_agg_feature(active, num_aggregations, 'ACT_')
    bureau_agg = bureau_agg.join(active_agg, how='left')
    del active, active_agg
    gc.collect()
    

    print("Bureau closed credit num feature aggregation ")
    # Bureau: Closed credits - using only numerical aggregations
    close = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = generate_agg_feature(close, num_aggregations, 'CLS_')
    bureau_agg = bureau_agg.join(closed_agg, how='left')
    del close, closed_agg, bureau
    gc.collect()
    return bureau_agg

def bureau_balance_month(bb):
    bb = generate_agg_feature(bb, {"NUM_STATUS": ["mean", "sum", "count"]}, "", ["SK_ID_CURR", "MONTHS_BALANCE"])
    bb = bb.reset_index()
    
    aggregation = {
        "MONTHS_BALANCE": ["count"],
        "NUM_STATUS_MEAN": ["mean", "max"],
        "NUM_STATUS_SUM": ["mean", "max"],
        "NUM_STATUS_COUNT": ["mean", "max", "min"]
    }
    
    res = generate_agg_feature(bb, aggregation, "BBM_")

    return res

def load_and_clean_previous_applications(num_rows = None, nan_as_category = True):
    print("Load dataset")
    prev = pd.read_csv('input/previous_application.csv', nrows = num_rows)
    
    print("Clean")
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    prev["NFLAG_LAST_APPL_IN_DAY"] = prev["NFLAG_LAST_APPL_IN_DAY"].astype("object")
    #prev["SELLERPLACE_AREA"] = prev["SELLERPLACE_AREA"].astype("object")
    prev["NFLAG_INSURED_ON_APPROVAL"] = prev["NFLAG_INSURED_ON_APPROVAL"].astype("object")
    
    good_cate = prev.groupby("NAME_GOODS_CATEGORY").SK_ID_PREV.count()
    good_cate = good_cate[good_cate < 1000].index
    prev.loc[prev.NAME_GOODS_CATEGORY.isin(good_cate), "NAME_GOODS_CATEGORY"] = "Other"

    prev.loc[prev.NAME_CONTRACT_TYPE.isin(["XNA", "XAP"]), "NAME_CONTRACT_TYPE"] = None
    prev.loc[prev.CODE_REJECT_REASON.isin(["XNA", "XAP"]), "CODE_REJECT_REASON"] = None
    prev.loc[prev.NAME_GOODS_CATEGORY.isin(["XNA", "XAP"]), "NAME_GOODS_CATEGORY"] = None
    
    prev["NB_PAYMENT"] = prev["AMT_CREDIT"] / prev["AMT_ANNUITY"]
    prev["NB_PAYMENT_INV"] = prev["AMT_ANNUITY"] / prev["AMT_CREDIT"]
    prev["CREDIT_GOOD_PERC"] = prev["AMT_CREDIT"] / prev["AMT_GOODS_PRICE"]
    
    prev["DEC_ST_DUE"] = prev["DAYS_FIRST_DUE"] - prev["DAYS_DECISION"]
    prev["DURING"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_DECISION"]
    prev["DURING_LAST_DUE"] = prev["DAYS_LAST_DUE"] - prev["DAYS_DECISION"]
    prev["DURING_TERM"] = prev["DAYS_TERMINATION"] - prev["DAYS_DECISION"]
    
    prev["LAST_DUE_1ST_VERSION_DIFF"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
    prev["LAST_DUE_1ST_VERSION_DIFF"] = prev["LAST_DUE_1ST_VERSION_DIFF"].apply(lambda x: x if x >0 else 0)
   
    prev["DAYS_TERMINATION_DIFF"] = prev["DAYS_TERMINATION"] - prev["DAYS_LAST_DUE"]
    prev["DAYS_TERMINATION_DIFF"] = prev["DAYS_TERMINATION_DIFF"].apply(lambda x: x if x >0 else 0)
    
    prev["DURING_LAST_DUE_PERC"] = prev["DURING_LAST_DUE"] / prev["DURING"]
    prev["DURING_TERM_PERC"] = prev["DURING_TERM"] / prev["DURING_LAST_DUE"]
    
    prev["PAYMENT_DURING"] = prev["CNT_PAYMENT"] / prev["DURING"]
    prev["CREDIT_DURING"] = prev["AMT_CREDIT"] / prev["DURING"]
    
    print("One hot encoder")
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    return prev, cat_cols

def recent_previous_feature_generation(prev, cat_cols):
    res = pd.DataFrame(index=prev.SK_ID_CURR.unique())

    num_aggregations = {
        'SK_ID_PREV': ['count'],
        'AMT_ANNUITY': ['mean'],
        'AMT_APPLICATION': ['mean'],
        'AMT_CREDIT': ['mean'],
        'APP_CREDIT_PERC': ['mean'],
        'AMT_DOWN_PAYMENT': ['mean'],
        'AMT_GOODS_PRICE': ['mean'],
        'HOUR_APPR_PROCESS_START': ['mean'],
        'RATE_DOWN_PAYMENT': ['mean'],
        'DAYS_DECISION': ['mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    }

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    print("Only num feature for instant")
    for time in [[0, 180], [180, 360], [360, 720]]:
        recent_prev = prev[(prev.DAYS_DECISION >= -time[1]) & (prev.DAYS_DECISION < time[0])].copy()
        recent_prev['APP_CREDIT_PERC'] = recent_prev['AMT_APPLICATION'] / recent_prev['AMT_CREDIT']
        
        prev_agg = recent_prev.groupby('SK_ID_CURR').agg({**num_aggregations})
        prev_agg.columns = pd.Index(['PREV_%d_' % time[0] + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        
        res = res.join(prev_agg, how="left")
 
    return res

def previous_application_feature_generation(prev, cat_cols):
    prev = prev.copy()

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['CAL_ANNUITY'] =  prev['AMT_CREDIT'] / prev['CNT_PAYMENT']
    prev["CAL_ANNUITY_PERC"] = prev['CAL_ANNUITY'] / prev["AMT_ANNUITY"]
    
    # Previous applications numeric features
    num_aggregations = {
        'SK_ID_PREV': ['count'],
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        "CAL_ANNUITY": ['min', 'max', 'mean'],
        "CAL_ANNUITY_PERC": ['min', 'max', 'mean'],
        "NB_PAYMENT": ["mean", "max", "min"],
        "NB_PAYMENT_INV": ["mean", "max", "min"],
        "CREDIT_GOOD_PERC": ["mean", "max", "min"],
    }
    
    prev_aggregations= {
        "DEC_ST_DUE": ["mean", "max"],
        "DURING": ["mean", "max", "min"],
        "DURING_LAST_DUE": ["mean", "max", "min"],
        "DURING_TERM": ["mean", "max", "min"],
        "DURING_LAST_DUE_PERC": ["mean"],
        "DURING_TERM_PERC": ["mean"],
        "PAYMENT_DURING": ["mean", "max", "min"],
        "CREDIT_DURING": ["mean", "max", "min"],
        "LAST_DUE_1ST_VERSION_DIFF": ["mean", "max"],
        "DAYS_TERMINATION_DIFF": ["mean"],
    }

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    print("Previous application feature / num feature aggregation")
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **prev_aggregations, **cat_aggregations})
#    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    print("Previous application approved num feature aggregation")
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = generate_agg_feature(approved, num_aggregations, 'APR_')
    prev_agg = prev_agg.join(approved_agg, how='left')
    del approved, approved_agg
    gc.collect()
    
    print("Previous application refused num feature aggregation")
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = generate_agg_feature(refused, num_aggregations, 'REF_')
    prev_agg = prev_agg.join(refused_agg, how='left')
    del refused, refused_agg
    gc.collect()
    
    print("On going application num feature")
    on_going = prev[(prev.NAME_CONTRACT_STATUS_Approved==1) & (prev.DAYS_TERMINATION.isnull())]
    on_going_agg = generate_agg_feature(on_going, num_aggregations, 'GOING_')
    prev_agg = prev_agg.join(on_going_agg, how='left')
    
    return prev_agg

def previous_application_complet(prev, pos, ins, cc):
    prev = prev.copy()
    pos = pos.copy()
    ins = ins.copy()
    cc = cc.copy()
    
    # pos
    agg_pos = generate_agg_feature(pos, {"CNT_INSTALMENT_FUTURE": ["min"]}, '', "SK_ID_PREV")
    prev = prev.join(agg_pos, on="SK_ID_PREV", how="left")
    prev["POS_FUTUR_AMT"] = prev["CNT_INSTALMENT_FUTURE_MIN"] * prev["AMT_ANNUITY"]
    prev["POS_PAYED_AMT"] = (prev['CNT_PAYMENT'] - prev["CNT_INSTALMENT_FUTURE_MIN"]) * prev["AMT_ANNUITY"]
    prev["POS_PAYED_AMT_PERC"] = prev["POS_PAYED_AMT"] / prev["AMT_CREDIT"]
    prev["POS_FUTUR_INS"] = prev["CNT_INSTALMENT_FUTURE_MIN"] / prev["CNT_PAYMENT"]
    #prev["POS_FUTUR_MON"] = prev[prev["CNT_INSTALMENT_FUTURE_MIN"] > 0]["AMT_ANNUITY"] 
    
    # ins
    ins_aggregations = {
        "AMT_PAYMENT": ["sum"],
        "NUM_INSTALMENT_NUMBER": ["max"]
    }
    agg_ins = generate_agg_feature(ins, ins_aggregations, '', "SK_ID_PREV")
    prev = prev.join(agg_ins, on="SK_ID_PREV", how="left")
    prev["INS_FUTUR_AMT"] = prev["AMT_CREDIT"] - prev["AMT_PAYMENT_SUM"] 
    prev["INS_FUTUR_AMT"] = prev["INS_FUTUR_AMT"].apply(lambda x: x if x > 0 else 0)
    prev["INS_PAYED_AMT_PERC"] = prev["AMT_PAYMENT_SUM"] / prev["AMT_CREDIT"]
    
    prev["AMT_MENSU"] = prev[(prev["CNT_INSTALMENT_FUTURE_MIN"] > 0) | 
                             (prev["NUM_INSTALMENT_NUMBER_MAX"] < prev["CNT_PAYMENT"])]["AMT_ANNUITY"]
    
    aggregations = {
        "POS_FUTUR_AMT": ["sum", "mean"],
        "POS_PAYED_AMT": ["mean"],
        "POS_PAYED_AMT_PERC": ["mean", "min", "max"],
        "POS_FUTUR_INS": ["sum"],
        "INS_FUTUR_AMT": ["sum", "mean"],
        "INS_PAYED_AMT_PERC": ["mean", "min", "max"],
        "AMT_MENSU": ["sum", "mean"]
    }

    prev_agg = generate_agg_feature(prev, aggregations, 'PREV_', "SK_ID_CURR")
    
    return prev_agg

def load_and_clean_pos_cash(num_rows = None, nan_as_category = True):
    print("Load dataset")
    pos = pd.read_csv('input/POS_CASH_balance.csv', nrows = num_rows)
    
    print("One hot encoder")
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)

    return pos, cat_cols

def recent_pos_feature_generation(pos, cat_cols):
    res = pd.DataFrame(index=pos.SK_ID_CURR.unique())

    aggregations = {
        'SK_DPD': ['mean'],
        'SK_DPD_DEF': ['mean'],
    }    
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    for time in [[0, 6], [6, 12], [12, 24]]:
        recent_pos = pos[(pos.MONTHS_BALANCE >= -time[1]) & (pos.MONTHS_BALANCE < time[0])].copy()
        pos_agg = generate_agg_feature(recent_pos, aggregations, 'POS_%d_' % time[0], "SK_ID_CURR")
        res = res.join(pos_agg, how="left")
    
    return res

def pos_cash_feature_generation(pos, cat_cols):
    pos = pos.copy()
    
    pos["PAY_PERC"] = 1 - pos['CNT_INSTALMENT_FUTURE'] / pos['CNT_INSTALMENT']
    # Features
    print("Get return dataframe")
    res = pos[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()

    
    print("Aggregation on SK_ID_PREV")
    aggregations = {
        "MONTHS_BALANCE": ["max", "min"],
        "CNT_INSTALMENT": ["max"],
        "CNT_INSTALMENT_FUTURE": ["min"],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'PAY_PERC': ["max"]
    }
    
    pos_agg = generate_agg_feature(pos, aggregations, 'PREV_', "SK_ID_PREV")
    res = res.join(pos_agg, on="SK_ID_PREV", how="left")
    res.drop('SK_ID_PREV', axis=1, inplace=True)

    
    print("Previous Aggregation on SK_ID_CURR")
    aggregations = {
        'PREV_MONTHS_BALANCE_MAX': ["min"],
        'PREV_MONTHS_BALANCE_MIN': ["max"],
        'PREV_CNT_INSTALMENT_FUTURE_MIN': ["max"],
        'PREV_PAY_PERC_MAX': ["min"]
    }
    for feat in res.columns:
        if feat not in ["SK_ID_CURR"]:
            aggregations[feat] = ["mean", "std"]
    res = generate_agg_feature(res, aggregations, 'POS_', "SK_ID_CURR")
    
    
    print("Aggregation on SK_ID_CURR")
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'CNT_INSTALMENT': ['max', 'min']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = generate_agg_feature(pos, aggregations, 'POS_', "SK_ID_CURR")
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    res = res.join(pos_agg, how="left")
    
    res["POS_REPATER_DPD"] = (pos[pos.SK_DPD > 0].groupby("SK_ID_CURR").SK_ID_PREV.count() > 1).astype(int)
    res["POS_REPATER_DPD"].fillna(0, inplace=True)
    res["POS_REPATER_DPD_DEF"] = (pos[pos.SK_DPD_DEF > 0].groupby("SK_ID_CURR").SK_ID_PREV.count() > 1).astype(int)
    res["POS_REPATER_DPD_DEF"].fillna(0, inplace=True)
    
    return res

def pos_balance_month(pos):
    aggregation = {
        "SK_DPD_DEF": ["mean", "sum"],
        "SK_DPD": ["mean", "sum"],
    }
    
    pos = generate_agg_feature(pos, aggregation, "", ["SK_ID_CURR", "MONTHS_BALANCE"])
    pos = pos.reset_index()
    
    aggregation = {
        "SK_DPD_DEF_SUM": ["max", "mean"],
        "SK_DPD_DEF_MEAN": ["max", "mean"],
        "SK_DPD_SUM": ["max", "mean"],
        "SK_DPD_MEAN": ["max", "mean"],
    }
    
    res = generate_agg_feature(pos, aggregation, "POSM_")
    
    return res

def load_and_clean_installments_payments(num_rows = None, nan_as_category = True):
    print("Load dataset")
    ins = pd.read_csv('input/installments_payments.csv', nrows = num_rows)
    
    print("One hot encoder")
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)

    #ins["AMT_PAYMENT"] = np.exp(ins["DAYS_ENTRY_PAYMENT"] / 1000) * ins["AMT_PAYMENT"]
    #ins['AMT_INSTALMENT'] = np.exp(ins["DAYS_ENTRY_PAYMENT"] / 1000) * ins["AMT_INSTALMENT"]

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    ins["DAYS_INSTALMENT_MONTH"] = ins.DAYS_INSTALMENT // 30

    return ins, cat_cols

def recent_installment_feature_generation(ins, cat_cols):
    res = pd.DataFrame(index=ins.SK_ID_CURR.unique())

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['mean'],
        'DBD': ['mean'],
        'PAYMENT_PERC': ['mean'],
        'PAYMENT_DIFF': ['mean'],
    }

    for time in [[0, 180], [180, 360], [360, 720]]:
        recent_ins = ins[(ins.DAYS_ENTRY_PAYMENT >= -time[1]) & (ins.DAYS_ENTRY_PAYMENT < time[0])].copy()
        ins_agg = generate_agg_feature(recent_ins, aggregations, 'INS_%d_' % time[0], "SK_ID_CURR")
        res = res.join(ins_agg, how="left")

    return res

def installment_feature_generation(ins, cat_cols):
    ins = ins.copy()
    
    print("Get return dataframe")
    res = ins[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()

    
    print("Aggregation on SK_ID_PREV")
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_INSTALMENT': ['min', 'max'],
        'AMT_INSTALMENT': ["sum"],
        'AMT_PAYMENT': ['sum'],
        'DPD': ["max"],
        "DBD": ["max"],
        "PAYMENT_DIFF": ["mean"]
    }
    
    ins_agg = generate_agg_feature(ins, aggregations, 'PREV_', "SK_ID_PREV")
    res = res.join(ins_agg, on="SK_ID_PREV", how="left")
    res.drop('SK_ID_PREV', axis=1, inplace=True)

    
    print("Statical SK_ID_PREV feature")
    res["PREV_DURING"] = res["PREV_DAYS_INSTALMENT_MAX"] - res["PREV_DAYS_INSTALMENT_MIN"]
    res["PREV_DURING"] = res["PREV_DURING"].apply(lambda x: x if x > 0 else 0)
    res["PREV_PAY_PERC"] = res["PREV_AMT_PAYMENT_SUM"] / res["PREV_AMT_INSTALMENT_SUM"]

    
    print("Previous Aggregation on SK_ID_CURR")
    aggregations = {
        'PREV_NUM_INSTALMENT_VERSION_NUNIQUE': ["max"],
        'PREV_AMT_INSTALMENT_SUM': ['max', 'min'],
        'PREV_AMT_PAYMENT_SUM': ['max', 'min'],
        'PREV_DAYS_INSTALMENT_MAX': ['min'],
        'PREV_DAYS_INSTALMENT_MIN': ['max'],
        'PREV_DPD_MAX': ["min"],
        'PREV_DBD_MAX': ["min"],
    }
    for feat in res.columns:
        if feat not in ["SK_ID_CURR"]:
            aggregations[feat] = ["mean", "std"]
    res = generate_agg_feature(res, aggregations, 'INS_', "SK_ID_CURR")
    

    print("Installment num / cate feature generation")
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INS_COUNT'] = ins.groupby('SK_ID_CURR').size()
    res = res.join(ins_agg, how="left")

    #inte = ins[ins.DPD > 0].copy()
    #inte["DAYS_ENTRY_PAYMENT_DIFF"] = inte.sort_values("DAYS_ENTRY_PAYMENT").groupby("SK_ID_CURR").DAYS_ENTRY_PAYMENT.diff()
    #aggregation = {
    #    "DAYS_ENTRY_PAYMENT_DIFF": ["min", "max", "mean"]
    #}
    #inte_agg = generate_agg_feature(inte, aggregation, 'PREV_')
    #res = res.join(inte_agg, how="left")
    
    del ins
    gc.collect()
    return res

def installment_balance_month(ins):
    aggregation = {
        "PAYMENT_DIFF": ["mean", "sum"],
        "DPD": ["mean", "sum"],
        "DBD": ["mean", "sum"],
        "AMT_INSTALMENT": ["mean", "sum", "count"],
        "AMT_PAYMENT": ["mean", "sum"],
    }
    
    ins = generate_agg_feature(ins, aggregation, "", ["SK_ID_CURR", "DAYS_INSTALMENT_MONTH"])
    ins = ins.reset_index()
    
    aggregation = {
        "DAYS_INSTALMENT_MONTH": ["count"],
        "AMT_INSTALMENT_SUM": ["nunique", "mean", "max", "min"],
        "AMT_INSTALMENT_MEAN": ["mean", "max", "min"],
        "AMT_INSTALMENT_COUNT": ["mean", "max", "min"],
        "AMT_PAYMENT_SUM": ["mean", "max", "min"],
        "AMT_PAYMENT_MEAN": ["mean", "max", "min"],
        "PAYMENT_DIFF_MEAN": ["mean", "max"],
        "PAYMENT_DIFF_SUM": ["mean", "max"],
        "DPD_MEAN": ["mean", "max"],
        "DPD_SUM": ["mean", "max"],
        "DBD_MEAN": ["mean", "max"],
        "DBD_SUM": ["mean", "max"],
    }
    
    res = generate_agg_feature(ins, aggregation, "INSM_")
    
    ins = ins[ins.DPD_SUM > 0]
    ins_agg = generate_agg_feature(ins, {"DAYS_INSTALMENT_MONTH": ["max"]}, "INSM_")
    res = res.join(ins_agg, how="left")
    
    return res

def load_and_clean_credit_card_balance(num_rows = None, nan_as_category = True):
    print("Load dataset")
    cc = pd.read_csv('input/credit_card_balance.csv', nrows = num_rows)
    
    cc.drop("NAME_CONTRACT_STATUS", axis=1, inplace=True)
    print("One hot encoder")
    #cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    
    return cc, []

def credit_card_feature_generation(cc, cat_cols):
    cc = cc.copy()
    
    print("Get return dataframe")
    res = cc[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()

    
    aggregations = {
        "CNT_DRAWINGS_CURRENT": ['mean', 'var'],
        "CNT_DRAWINGS_ATM_CURRENT": ["mean", 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL': ["mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["mean", "sum"],
        "AMT_PAYMENT_CURRENT": ["mean", "sum"],
        "AMT_DRAWINGS_ATM_CURRENT": ["mean", "sum"],
        "AMT_INST_MIN_REGULARITY": ["var"]
    }
    print("Aggregation on SK_ID_PREV")
    cc_agg = generate_agg_feature(cc, aggregations, 'PREV_', "SK_ID_PREV")
    res = res.join(cc_agg, on="SK_ID_PREV", how="left")
    res.drop('SK_ID_PREV', axis=1, inplace=True)

    
    print("Previous Aggregation on SK_ID_CURR")
    aggregations = {

    }
    for feat in res.columns:
        if feat not in ["SK_ID_CURR"]:
            aggregations[feat] = ["mean"]
    res = generate_agg_feature(res, aggregations, 'CC_', "SK_ID_CURR")
    
    
    print("Aggregation on SK_ID_CURR")
    aggregations = {
        "CNT_DRAWINGS_ATM_CURRENT": ["mean"],
        "CNT_DRAWINGS_CURRENT": ["mean", "var"],
        "AMT_DRAWINGS_ATM_CURRENT": ["sum", "mean"],
        "AMT_INST_MIN_REGULARITY": ["var"],
        "AMT_PAYMENT_CURRENT": ["sum", "min", "mean"],
        "AMT_DRAWINGS_POS_CURRENT": ["sum"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["mean"]
    }
    # General aggregations
    cc.drop(['SK_ID_PREV'], inplace = True, axis=1)
    cc_agg = generate_agg_feature(cc, aggregations, 'CC_', "SK_ID_CURR")
    res = res.join(cc_agg, how="left")
    
    del cc
    gc.collect()
    return res


def global_feature(dataset):
    dataset = dataset.copy()
    
    dataset["AMT_ANNUITY_PREV_PERC"] = dataset["AMT_ANNUITY"] / dataset["PREV_AMT_ANNUITY_MEAN"]
    dataset["AMT_CREDIT_PREV_PERC"] = dataset["AMT_CREDIT"] / dataset["PREV_AMT_CREDIT_MEAN"]
    dataset["AMT_GOODS_PRICE_PREV_PERC"] = dataset["AMT_GOODS_PRICE"] / dataset["PREV_AMT_GOODS_PRICE_MEAN"]
    
    dataset["PREV_ANNUITY_DIV_INCOME"] = dataset["PREV_AMT_ANNUITY_MEAN"] / dataset["AMT_INCOME_TOTAL"]
    dataset["PREV_CREDIT_DIV_INCOME"] = dataset["PREV_AMT_CREDIT_MEAN"] / dataset["AMT_INCOME_TOTAL"]
    dataset["PREV_GOODS_DIV_INCOME"] = dataset["PREV_AMT_GOODS_PRICE_MEAN"] / dataset["AMT_INCOME_TOTAL"]
    
    dataset["ANNUITY_DIV_BURO_CREDIT_SUM"] = dataset["AMT_ANNUITY"] / dataset["BURO_AMT_CREDIT_SUM_MEAN"]
    dataset["CREDIT_DIV_BURO_CREDIT_SUM"] = dataset["AMT_CREDIT"] / dataset["BURO_AMT_CREDIT_SUM_MEAN"]
    dataset["GOODS_DIV_BURO_CREDIT_SUM"] = dataset["AMT_GOODS_PRICE"] / dataset["BURO_AMT_CREDIT_SUM_MEAN"]
    dataset["INCOME_DIV_BURO_CREDIT_SUM"] = dataset["AMT_INCOME_TOTAL"] / dataset["BURO_AMT_CREDIT_SUM_MEAN"]

    return dataset


def pickle_load(dataset, path):
    with open(path, "rb") as file:
        return pickle.load(file)
    dataset = filter_weak_feature(dataset, 22)
    print(dataset.shape)
    return dataset

def prepare_model_base_2():
    application, cate_feature = load_and_clean_application_train_test()
    bureau, bb, bureau_cat, bb_cat = load_and_clean_bureau_and_balance()
    prev, prev_cat_cols = load_and_clean_previous_applications()
    pos, pos_cat_cols = load_and_clean_pos_cash()
    ins, ins_cat_cols = load_and_clean_installments_payments()
    cc, cc_cat_cols = load_and_clean_credit_card_balance()
    df = application_feature_generation(application, cate_feature)
    bureau_feature = bureau_feature_generation(bureau, bureau_cat, bb_cat)
    recent_bureau_feature = recent_bureau_feature_generation()
    prev_feature = previous_application_feature_generation(prev, prev_cat_cols)
    recent_prev_feature = recent_previous_feature_generation(prev, prev_cat_cols)
    pos_cash_feature = pos_cash_feature_generation(pos, pos_cat_cols)
    recent_pos_cash_feature = recent_pos_feature_generation(pos, pos_cat_cols)
    installment_feature = installment_feature_generation(ins, ins_cat_cols)
    recent_installment_feature = recent_installment_feature_generation(ins, ins_cat_cols)
    cd_feature = credit_card_feature_generation(cc, cc_cat_cols)
    prev_agg = previous_application_complet(prev, pos, ins, cc)
    insm = installment_balance_month(ins)
    posm = pos_balance_month(pos)
    bbm = bureau_balance_month(bb)
    dataset = df.join(
        bureau_feature).join(
        prev_feature).join(
        pos_cash_feature).join(
        installment_feature).join(
        cd_feature).join(
        recent_bureau_feature).join(
        recent_prev_feature).join(
        recent_pos_cash_feature).join(
        recent_installment_feature).join(
        prev_agg).join(
        insm).join(
        posm).join(
        bbm)
    dataset = global_feature(dataset)
    filter_weak_feature(dataset, 22)
    train_df = dataset[dataset['TARGET'].notnull()]
    test_df = dataset[dataset['TARGET'].isnull()]
    test_df = test_df.drop(columns = ["TARGET"])
    print(train_df.shape)
    train_df.to_csv("processed/train_model_base_2.csv")
    test_df.to_csv("processed/test_model_base_2.csv")
    