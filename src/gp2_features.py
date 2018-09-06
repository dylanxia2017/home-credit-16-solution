import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

def GP2(data):
    v = pd.DataFrame()
    v["i0"] = 0.059995*np.tanh(((((((((((data["DAYS_BIRTH"]) + (data["REGION_RATING_CLIENT_W_CITY"]))) * 2.0)) + (((np.where(data["REFUSED_AMT_CREDIT_MAX"] < -99998, -1.0, 3.141593 )) + (data["NAME_INCOME_TYPE_Working"]))))) + (data["DAYS_BIRTH"]))) * 2.0)) 
    v["i1"] = 0.095000*np.tanh(((((((((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]<0, np.maximum(((data["REGION_RATING_CLIENT_W_CITY"])), ((((1.0) + (data["REFUSED_DAYS_DECISION_MEAN"]))))), data["REFUSED_DAYS_DECISION_MEAN"] )) - (data["CODE_GENDER"]))) + (data["DAYS_BIRTH"]))) * 2.0)) * 2.0)) 
    v["i2"] = 0.098800*np.tanh((((((((((-1.0*((np.tanh(((((((1.0)) / 2.0)) - (data["CLOSED_DAYS_CREDIT_MEAN"])))))))) - (data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) * 2.0)) - (((0.636620) - (data["CLOSED_DAYS_CREDIT_MEAN"]))))) * 2.0)) 
    v["i3"] = 0.097103*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((((np.tanh((data["REFUSED_DAYS_DECISION_MAX"]))) + (((data["REGION_RATING_CLIENT"]) + (((np.maximum(((((data["DAYS_BIRTH"]) - (data["NAME_EDUCATION_TYPE_Higher_education"])))), ((data["REFUSED_CNT_PAYMENT_SUM"])))) * 2.0)))))) * 2.0)))) 
    v["i4"] = 0.099996*np.tanh(((np.tanh((data["REFUSED_DAYS_DECISION_MEAN"]))) + (((np.maximum(((((data["DAYS_ID_PUBLISH"]) + (data["NAME_INCOME_TYPE_Working"])))), ((data["REFUSED_CNT_PAYMENT_SUM"])))) + (((((data["DAYS_BIRTH"]) - (data["CODE_GENDER"]))) + (data["REGION_RATING_CLIENT_W_CITY"]))))))) 
    v["i5"] = 0.099985*np.tanh(((((np.maximum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])), ((((((np.where(data["NEW_SCORES_STD"]<0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], data["NEW_DOC_IND_KURT"] )) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) + (np.maximum(((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])), ((data["NEW_SCORES_STD"]))))))))) * 2.0)) * 2.0)) 
    v["i6"] = 0.099930*np.tanh(((((((((((data["REG_CITY_NOT_LIVE_CITY"]) / 2.0)) + (data["REGION_RATING_CLIENT_W_CITY"]))) + (((((data["FLAG_DOCUMENT_3"]) + (data["DAYS_ID_PUBLISH"]))) * 2.0)))) + (((data["REG_CITY_NOT_WORK_CITY"]) - (data["EMERGENCYSTATE_MODE_No"]))))) * 2.0)) 
    v["i7"] = 0.099976*np.tanh((((7.0)) * (np.where(data["CODE_GENDER"]<0, 3.0, ((((((((data["REFUSED_AMT_APPLICATION_MAX"]) < (data["REFUSED_CNT_PAYMENT_SUM"]))*1.)) + (((data["REFUSED_CNT_PAYMENT_SUM"]) * 2.0)))) + (((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0)))/2.0) )))) 
    v["i8"] = 0.099595*np.tanh(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, data["FLAG_OWN_CAR"], np.where(np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["NAME_EDUCATION_TYPE_Higher_education"], data["NEW_CAR_TO_BIRTH_RATIO"] )>0, (3.0), ((data["CLOSED_DAYS_CREDIT_VAR"]) - ((11.01282405853271484))) ) )))) 
    v["i9"] = 0.099984*np.tanh(((((data["REG_CITY_NOT_LIVE_CITY"]) + (np.where(data["NEW_DOC_IND_KURT"]>0, ((np.minimum(((data["REG_CITY_NOT_LIVE_CITY"])), ((data["ACTIVE_DAYS_CREDIT_VAR"])))) - (data["NEW_CAR_TO_BIRTH_RATIO"])), -2.0 )))) + ((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["NEW_DOC_IND_KURT"]))/2.0)))) 
    v["i10"] = 0.100000*np.tanh(((((((data["NEW_SCORES_STD"]) - (np.where((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) + ((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) + (np.tanh((1.0))))/2.0)))/2.0)>0, (1.0), data["NAME_FAMILY_STATUS_Married"] )))) * 2.0)) - (data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) 
    v["i11"] = 0.099915*np.tanh((((((((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (data["EXT_SOURCE_1"]))*1.)) - (((np.where(data["EXT_SOURCE_1"] < -99998, np.tanh((data["ACTIVE_DAYS_CREDIT_VAR"])), data["EXT_SOURCE_1"] )) - (np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) * 2.0)) * 2.0)) 
    v["i12"] = 0.099796*np.tanh(((((data["DAYS_BIRTH"]) - (data["NAME_INCOME_TYPE_State_servant"]))) + ((((((data["DAYS_REGISTRATION"]) + (np.where(data["NAME_FAMILY_STATUS_Married"]>0, (((data["ACTIVE_DAYS_CREDIT_VAR"]) + (2.0))/2.0), 1.570796 )))/2.0)) - (data["WALLSMATERIAL_MODE_Panel"]))))) 
    v["i13"] = 0.100000*np.tanh((((((((-1.0*((((((np.maximum(((data["EXT_SOURCE_1"])), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))) * 2.0)) * 2.0))))) - (np.maximum(((((data["OCCUPATION_TYPE_Core_staff"]) / 2.0))), ((0.318310)))))) * 2.0)) * 2.0)) 
    v["i14"] = 0.099680*np.tanh(((data["NEW_DOC_IND_KURT"]) + (((((((data["NEW_DOC_IND_KURT"]) + (np.tanh((np.where(data["REFUSED_AMT_CREDIT_MEAN"]>0, data["NEW_DOC_IND_KURT"], ((data["REFUSED_DAYS_DECISION_MEAN"]) * 2.0) )))))) - (data["REFUSED_AMT_CREDIT_MEAN"]))) + (data["REFUSED_CNT_PAYMENT_SUM"]))))) 
    v["i15"] = 0.099997*np.tanh(((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (data["REG_CITY_NOT_LIVE_CITY"]))) - (data["FLAG_PHONE"]))) + (np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["DAYS_REGISTRATION"], ((data["ORGANIZATION_TYPE_Self_employed"]) + (((data["NEW_SCORES_STD"]) + (data["REG_CITY_NOT_LIVE_CITY"])))) )))) 
    v["i16"] = 0.099925*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, 3.141593, ((data["REGION_RATING_CLIENT"]) - (np.where((((np.tanh((data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))*1.)>0, (4.07468795776367188), data["APPROVED_AMT_DOWN_PAYMENT_MAX"] ))) )) 
    v["i17"] = 0.099600*np.tanh(((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((data["ORGANIZATION_TYPE_Transport__type_3"])))) + ((((np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, ((data["FLAG_WORK_PHONE"]) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"])))), (13.65371227264404297) )) + (data["ORGANIZATION_TYPE_Construction"]))/2.0)))) 
    v["i18"] = 0.099990*np.tanh((-1.0*((np.where(((np.maximum(((np.maximum(((data["OCCUPATION_TYPE_Accountants"])), ((data["EXT_SOURCE_1"]))))), ((np.maximum(((data["NEW_CAR_TO_EMPLOY_RATIO"])), ((data["NEW_CAR_TO_EMPLOY_RATIO"]))))))) + (data["OCCUPATION_TYPE_High_skill_tech_staff"]))<0, data["EXT_SOURCE_1"], (13.81985569000244141) ))))) 
    v["i19"] = 0.095400*np.tanh(((((data["OCCUPATION_TYPE_Drivers"]) - (data["OCCUPATION_TYPE_Core_staff"]))) + (np.maximum(((data["ACTIVE_AMT_ANNUITY_MAX"])), ((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, -1.0, (-1.0*((data["ACTIVE_DAYS_CREDIT_VAR"]))) ))))))) 
    v["i20"] = 0.099600*np.tanh(np.where((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["NAME_INCOME_TYPE_Unemployed"]))*1.)>0, (9.65223503112792969), np.where(((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (data["DAYS_ID_PUBLISH"])) < -99998, np.maximum(((data["DAYS_ID_PUBLISH"])), ((data["NEW_DOC_IND_KURT"]))), -2.0 ) )) 
    v["i21"] = 0.099870*np.tanh(((data["INSTAL_AMT_INSTALMENT_MAX"]) + (((((((((np.where((((data["INSTAL_AMT_INSTALMENT_MAX"]) + (((1.0) / 2.0)))/2.0)>0, data["INSTAL_AMT_INSTALMENT_MAX"], 1.570796 )) * 2.0)) * 2.0)) * 2.0)) - (1.570796))))) 
    v["i22"] = 0.099732*np.tanh(((np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] < -99998, np.where(((((0.318310) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (data["NEW_DOC_IND_KURT"]))>0, 0.318310, data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] ), data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] ))))) * 2.0)) 
    v["i23"] = 0.099959*np.tanh((((-1.0*((((np.where(data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"] < -99998, data["ORGANIZATION_TYPE_Military"], -1.0 )) * 2.0))))) - (((((((((((-1.0) + (data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]))/2.0)) < (data["CC_MONTHS_BALANCE_VAR"]))*1.)) * 2.0)) * 2.0)))) 
    v["i24"] = 0.099950*np.tanh(np.maximum(((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])))), ((np.where((((data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]) < (((1.570796) * (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]))))*1.)>0, 3.141593, (-1.0*((0.318310))) ))))) 
    v["i25"] = 0.099964*np.tanh(np.where(np.maximum(((data["NAME_EDUCATION_TYPE_Lower_secondary"])), (((((((0.318310) / 2.0)) > ((-1.0*((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))))*1.))))>0, (11.36913585662841797), (-1.0*(((((((data["NAME_FAMILY_STATUS_Married"]) > (data["NAME_EDUCATION_TYPE_Lower_secondary"]))*1.)) / 2.0)))) )) 
    v["i26"] = 0.097999*np.tanh(np.minimum(((((((((1.570796) > (((0.318310) + (data["HOUR_APPR_PROCESS_START"]))))*1.)) + (data["OCCUPATION_TYPE_Laborers"]))/2.0))), (((-1.0*((((data["CC_NAME_CONTRACT_STATUS_Approved_VAR"]) + (data["HOUR_APPR_PROCESS_START"]))))))))) 
    v["i27"] = 0.099000*np.tanh(((((((((np.where((((data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]) < (((np.tanh((np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))*1.)>0, 2.0, data["NAME_INCOME_TYPE_Unemployed"] )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i28"] = 0.097996*np.tanh((((((np.maximum(((data["OCCUPATION_TYPE_Low_skill_Laborers"])), (((((((np.maximum(((data["ORGANIZATION_TYPE_Realtor"])), ((data["NAME_INCOME_TYPE_Unemployed"])))) + (data["ORGANIZATION_TYPE_Transport__type_3"]))/2.0)) + (data["NAME_INCOME_TYPE_Maternity_leave"])))))) + (data["ORGANIZATION_TYPE_Trade__type_3"]))/2.0)) - (data["ORGANIZATION_TYPE_Industry__type_9"]))) 
    v["i29"] = 0.098000*np.tanh(((((np.tanh((np.tanh(((((data["ORGANIZATION_TYPE_Self_employed"]) + (np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]))))/2.0)))))) - (np.maximum(((((data["ORGANIZATION_TYPE_Military"]) * 2.0))), ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))))) - (np.tanh((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))))) 
    v["i30"] = 0.097500*np.tanh(((((((data["ORGANIZATION_TYPE_Realtor"]) + (((data["NAME_INCOME_TYPE_Unemployed"]) + (data["ORGANIZATION_TYPE_Restaurant"]))))) - (np.maximum(((data["ORGANIZATION_TYPE_Trade__type_2"])), ((data["OCCUPATION_TYPE_Medicine_staff"])))))) - (np.maximum(((data["ORGANIZATION_TYPE_Trade__type_2"])), ((data["ORGANIZATION_TYPE_Security_Ministries"])))))) 
    v["i31"] = 0.098960*np.tanh(((((((data["NAME_INCOME_TYPE_Unemployed"]) + ((((data["REFUSED_DAYS_DECISION_MEAN"]) < (np.where(data["REFUSED_DAYS_DECISION_MAX"]<0, data["REFUSED_DAYS_DECISION_MAX"], (((np.tanh((data["REFUSED_DAYS_DECISION_MEAN"]))) < (data["REFUSED_DAYS_DECISION_MAX"]))*1.) )))*1.)))) * ((10.95812892913818359)))) * 2.0)) 
    v["i32"] = 0.099500*np.tanh(((((((((data["NAME_HOUSING_TYPE_Municipal_apartment"]) + (-2.0))/2.0)) * (np.maximum(((np.maximum(((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])), ((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]))))), ((data["NAME_HOUSING_TYPE_Municipal_apartment"])))))) + ((-1.0*(((((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) > (data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]))*1.))))))/2.0)) 
    v["i33"] = 0.096799*np.tanh(np.maximum((((((0.636620) + (np.maximum(((-1.0)), ((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]<0, data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"], data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] ))))))/2.0))), ((((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (-1.0))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])))))) 
    v["i34"] = 0.099997*np.tanh(np.minimum(((((np.maximum(((data["NEW_SCORES_STD"])), ((data["DAYS_REGISTRATION"])))) / 2.0))), ((((((((0.636620) - ((-1.0*((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (0.636620))) * 2.0))))))) * 2.0)) * 2.0))))) 
    v["i35"] = 0.099015*np.tanh(((np.minimum(((np.maximum(((((((data["NAME_EDUCATION_TYPE_Incomplete_higher"]) * (data["AMT_INCOME_TOTAL"]))) * 2.0))), ((data["NAME_EDUCATION_TYPE_Incomplete_higher"]))))), ((data["AMT_INCOME_TOTAL"])))) - (((((((data["NAME_EDUCATION_TYPE_Incomplete_higher"]) + (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]))/2.0)) + (data["ORGANIZATION_TYPE_Bank"]))/2.0)))) 
    v["i36"] = 0.098830*np.tanh(np.minimum(((1.570796)), (((((((((-1.0*((data["AMT_CREDIT"])))) * (((((((data["AMT_CREDIT"]) * 2.0)) * 2.0)) - (data["ORGANIZATION_TYPE_Military"]))))) - (data["AMT_CREDIT"]))) + (1.570796)))))) 
    v["i37"] = 0.098020*np.tanh((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (((((3.0) - (data["NAME_INCOME_TYPE_Unemployed"]))) / 2.0)))*1.)) - (((((((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) + ((-1.0*((-1.0)))))/2.0)) > (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)))) 
    v["i38"] = 0.099905*np.tanh(((np.maximum(((np.maximum((((((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]) < (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))*1.))), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))), (((((data["AMT_INCOME_TOTAL"]) > ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) < (np.tanh((((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) * 2.0)))))*1.)))*1.))))) * 2.0)) 
    v["i39"] = 0.097452*np.tanh((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((((-1.0*((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"])))) + (((((((data["ORGANIZATION_TYPE_Realtor"]) + (data["NAME_INCOME_TYPE_Unemployed"]))) + (data["ORGANIZATION_TYPE_Transport__type_3"]))) - (data["ORGANIZATION_TYPE_Military"]))))/2.0)))) + (data["ORGANIZATION_TYPE_Transport__type_3"]))/2.0)) 
    v["i40"] = 0.098003*np.tanh((((((((-1.0*((data["WEEKDAY_APPR_PROCESS_START_MONDAY"])))) + (((np.where(data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"] < -99998, data["ORGANIZATION_TYPE_Construction"], (10.0) )) * (data["ORGANIZATION_TYPE_Construction"]))))/2.0)) + ((-1.0*((np.maximum(((data["ORGANIZATION_TYPE_Police"])), ((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]))))))))/2.0)) 
    v["i41"] = 0.100000*np.tanh(np.where(data["PREV_AMT_ANNUITY_MEAN"] < -99998, np.tanh((data["PREV_APP_CREDIT_PERC_MEAN"])), ((data["AMT_INCOME_TOTAL"]) - (((data["PREV_AMT_ANNUITY_MEAN"]) + (np.tanh((((data["PREV_APP_CREDIT_PERC_MEAN"]) * (((3.141593) + (data["PREV_APP_CREDIT_PERC_MAX"])))))))))) )) 
    v["i42"] = 0.099499*np.tanh((((((((((data["OCCUPATION_TYPE_Sales_staff"]) + (data["NAME_INCOME_TYPE_Maternity_leave"]))/2.0)) + (((data["ORGANIZATION_TYPE_Trade__type_7"]) - (data["ORGANIZATION_TYPE_Industry__type_12"]))))/2.0)) + (((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["OCCUPATION_TYPE_Cooking_staff"]))) - (data["ORGANIZATION_TYPE_Industry__type_9"]))))/2.0)) 
    v["i43"] = 0.098011*np.tanh(np.where(((data["ORGANIZATION_TYPE_School"]) * (data["CODE_GENDER"]))>0, ((np.tanh((((data["CODE_GENDER"]) * (data["AMT_INCOME_TOTAL"]))))) - (data["ORGANIZATION_TYPE_School"])), (((data["OCCUPATION_TYPE_Security_staff"]) + (data["AMT_INCOME_TOTAL"]))/2.0) )) 
    v["i44"] = 0.095501*np.tanh((((((((data["ORGANIZATION_TYPE_Realtor"]) > ((-1.0*((((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"] < -99998, data["OCCUPATION_TYPE_Cleaning_staff"], data["APPROVED_AMT_DOWN_PAYMENT_MEAN"] )) - (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))))))*1.)) + (((data["OCCUPATION_TYPE_Cleaning_staff"]) * 2.0)))) + (data["OCCUPATION_TYPE_Cleaning_staff"]))) 
    v["i45"] = 0.099831*np.tanh(((data["NAME_INCOME_TYPE_Unemployed"]) + (((((data["NAME_HOUSING_TYPE_Office_apartment"]) * (data["REG_REGION_NOT_LIVE_REGION"]))) + (((data["HOUSETYPE_MODE_specific_housing"]) + (((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (((data["AMT_INCOME_TOTAL"]) * 2.0)))))))))) 
    v["i46"] = 0.099521*np.tanh((-1.0*((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, np.maximum(((data["FONDKAPREMONT_MODE_org_spec_account"])), ((data["ORGANIZATION_TYPE_Security_Ministries"]))), np.where((((1.0) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)<0, data["FONDKAPREMONT_MODE_org_spec_account"], ((data["NEW_CAR_TO_EMPLOY_RATIO"]) * 2.0) ) )) * 2.0))))) 
    v["i47"] = 0.099301*np.tanh(((((3.0) * (np.where(data["ORGANIZATION_TYPE_Realtor"]>0, 1.0, np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, data["ORGANIZATION_TYPE_Legal_Services"], ((((((7.0)) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))) + (2.0))/2.0) ) )))) * 2.0)) 
    v["i48"] = 0.098000*np.tanh((-1.0*((np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, ((((3.141593) - (data["AMT_REQ_CREDIT_BUREAU_QRT"]))) - (data["AMT_REQ_CREDIT_BUREAU_QRT"])), np.tanh((np.tanh(((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + ((-1.0*((data["NEW_CREDIT_TO_INCOME_RATIO"])))))/2.0))))) ))))) 
    v["i49"] = 0.099600*np.tanh(((((-1.0) + ((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) + (data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]))/2.0)))) + ((((((((((data["POS_SK_DPD_MAX"]) > (((data["NAME_INCOME_TYPE_Unemployed"]) + (data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]))))*1.)) * 2.0)) * 2.0)) * 2.0)))) 
    v["i50"] = 0.095100*np.tanh(((np.minimum((((((data["AMT_CREDIT"]) > (-1.0))*1.))), (((((data["AMT_CREDIT"]) < (np.where(data["AMT_CREDIT"]<0, data["NAME_TYPE_SUITE_Other_B"], ((0.318310) / 2.0) )))*1.))))) - (0.318310))) 
    v["i51"] = 0.099453*np.tanh(((((((np.where((((data["NAME_INCOME_TYPE_Unemployed"]) < (((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) / 2.0)))*1.)>0, 1.570796, ((((data["NAME_INCOME_TYPE_Unemployed"]) * 2.0)) * 2.0) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i52"] = 0.099700*np.tanh(((np.where(data["AMT_INCOME_TOTAL"]<0, (-1.0*((data["AMT_INCOME_TOTAL"]))), ((np.minimum((((-1.0*((np.minimum(((data["AMT_INCOME_TOTAL"])), ((data["BURO_DAYS_CREDIT_VAR"])))))))), ((data["BURO_DAYS_CREDIT_VAR"])))) / 2.0) )) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))) 
    v["i53"] = 0.099200*np.tanh((((((((data["NAME_HOUSING_TYPE_Rented_apartment"]) - (data["ORGANIZATION_TYPE_Industry__type_12"]))) - ((((data["NAME_FAMILY_STATUS_Married"]) + (np.where(data["NAME_FAMILY_STATUS_Married"]>0, data["ORGANIZATION_TYPE_Transport__type_2"], data["NAME_INCOME_TYPE_Unemployed"] )))/2.0)))) + ((-1.0*((((data["FLAG_OWN_REALTY"]) / 2.0))))))/2.0)) 
    v["i54"] = 0.090218*np.tanh(((((data["FLAG_WORK_PHONE"]) * ((((((((data["ORGANIZATION_TYPE_Hotel"]) - (data["AMT_INCOME_TOTAL"]))) > ((((data["AMT_INCOME_TOTAL"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)))*1.)) - ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["AMT_INCOME_TOTAL"]))/2.0)))))) / 2.0)) 
    v["i55"] = 0.099294*np.tanh(np.where(data["ORGANIZATION_TYPE_XNA"]<0, (((((data["NAME_EDUCATION_TYPE_Academic_degree"]) > (((((2.0) * (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))) + (0.318310))))*1.)) - (0.318310)), (((data["NAME_EDUCATION_TYPE_Academic_degree"]) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.) )) 
    v["i56"] = 0.086898*np.tanh(np.where(data["ORGANIZATION_TYPE_Transport__type_4"]>0, data["ORGANIZATION_TYPE_Transport__type_4"], np.where(data["AMT_INCOME_TOTAL"]>0, ((data["AMT_INCOME_TOTAL"]) * (((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) - (data["AMT_INCOME_TOTAL"])))), ((data["ORGANIZATION_TYPE_Transport__type_4"]) - (data["NAME_FAMILY_STATUS_Widow"])) ) )) 
    v["i57"] = 0.088497*np.tanh((((((data["ORGANIZATION_TYPE_Trade__type_3"]) + ((((((((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_TYPE_SUITE_Spouse__partner"]))/2.0)) + (data["NAME_INCOME_TYPE_Maternity_leave"]))/2.0)) + ((-1.0*((data["NAME_TYPE_SUITE_Spouse__partner"])))))/2.0)))/2.0)) - (np.maximum(((data["NAME_EDUCATION_TYPE_Academic_degree"])), ((data["ORGANIZATION_TYPE_Hotel"])))))) 
    v["i58"] = 0.095795*np.tanh((((((data["NAME_INCOME_TYPE_Unemployed"]) + (((((data["OCCUPATION_TYPE_Waiters_barmen_staff"]) / 2.0)) - (data["REG_REGION_NOT_LIVE_REGION"]))))/2.0)) + (((data["NAME_TYPE_SUITE_Other_B"]) + ((((data["NAME_TYPE_SUITE_Other_B"]) > (((data["REG_REGION_NOT_LIVE_REGION"]) - (data["CLOSED_AMT_ANNUITY_MEAN"]))))*1.)))))) 
    v["i59"] = 0.097340*np.tanh(((np.maximum(((((((np.where(data["BURO_STATUS_2_MEAN_MEAN"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], data["BURO_STATUS_4_MEAN_MEAN"] )) * 2.0)) * 2.0))), ((((((np.maximum(((data["BURO_MONTHS_BALANCE_MIN_MIN"])), ((data["BURO_STATUS_2_MEAN_MEAN"])))) * 2.0)) - (3.0)))))) * 2.0)) 
    v["i60"] = 0.098002*np.tanh((((((((((((data["ORGANIZATION_TYPE_Transport__type_3"]) * 2.0)) < (data["AMT_INCOME_TOTAL"]))*1.)) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) - (data["ORGANIZATION_TYPE_Electricity"]))))/2.0)) + (np.maximum(((data["ORGANIZATION_TYPE_Legal_Services"])), (((((data["ORGANIZATION_TYPE_Legal_Services"]) < (data["CLOSED_AMT_ANNUITY_MAX"]))*1.))))))/2.0)) 
    v["i61"] = 0.099951*np.tanh((((((data["ORGANIZATION_TYPE_Industry__type_3"]) - (data["FLAG_CONT_MOBILE"]))) + ((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_CONTRACT_TYPE_Revolving_loans"]))/2.0)))/2.0)) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))))/2.0)) 
    v["i62"] = 0.100000*np.tanh(((((np.where(data["OCCUPATION_TYPE_Accountants"]<0, data["ORGANIZATION_TYPE_Police"], data["OCCUPATION_TYPE_Accountants"] )) * (((data["FONDKAPREMONT_MODE_reg_oper_spec_account"]) + (((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["ORGANIZATION_TYPE_Police"]))))))) + (((data["OCCUPATION_TYPE_Low_skill_Laborers"]) - (data["WALLSMATERIAL_MODE_Monolithic"]))))) 
    v["i63"] = 0.095961*np.tanh(((np.maximum(((data["NAME_INCOME_TYPE_Unemployed"])), ((((np.maximum((((((data["OCCUPATION_TYPE_Security_staff"]) > (((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (data["ORGANIZATION_TYPE_Realtor"]))))*1.))), ((np.where(data["ORGANIZATION_TYPE_Realtor"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NEW_CREDIT_TO_INCOME_RATIO"] ))))) * 2.0))))) * 2.0)) 
    v["i64"] = 0.099960*np.tanh(((((((np.where(((((data["NEW_SOURCES_PROD"]) + (data["NEW_SOURCES_PROD"]))) + (1.570796))>0, -2.0, np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (10.0) ) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i65"] = 0.099910*np.tanh(np.where(((data["NEW_SOURCES_PROD"]) - (np.tanh((((data["EXT_SOURCE_1"]) * 2.0)))))>0, -2.0, (-1.0*((np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_EDUCATION_TYPE_Academic_degree"], ((((data["NEW_SOURCES_PROD"]) * 2.0)) * 2.0) )))) )) 
    v["i66"] = 0.099640*np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((((((((-1.0) - (((data["NEW_SOURCES_PROD"]) * 2.0)))) - (data["NEW_SCORES_STD"]))) * 2.0)) - (np.minimum(((data["NEW_SOURCES_PROD"])), ((data["NEW_SCORES_STD"]))))) )) 
    v["i67"] = 0.099950*np.tanh((-1.0*(((((((data["NEW_SOURCES_PROD"]) < (((data["REFUSED_AMT_GOODS_PRICE_MAX"]) + (1.570796))))*1.)) + (np.where(data["NEW_SOURCES_PROD"] < -99998, np.tanh((-2.0)), ((data["NEW_SOURCES_PROD"]) * 2.0) ))))))) 
    v["i68"] = 0.099490*np.tanh((-1.0*((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((np.where(data["NEW_SCORES_STD"]>0, data["DAYS_BIRTH"], (-1.0*(((((((((((((data["DAYS_BIRTH"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) > (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.)))) )))))))) 
    v["i69"] = 0.099500*np.tanh(np.where(((((((np.maximum(((1.570796)), ((data["CLOSED_AMT_CREDIT_SUM_MEAN"])))) < (data["CLOSED_DAYS_CREDIT_MEAN"]))*1.)) + (data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]))/2.0)>0, 3.0, (((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) > (((1.0) + (data["CLOSED_DAYS_CREDIT_MEAN"]))))*1.) )) 
    v["i70"] = 0.096400*np.tanh(np.where((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["DAYS_BIRTH"]))/2.0)>0, ((np.where(data["DAYS_BIRTH"]>0, ((2.0) - (data["DAYS_BIRTH"])), np.tanh((data["DAYS_BIRTH"])) )) - (data["DAYS_BIRTH"])), data["CLOSED_DAYS_CREDIT_MEAN"] )) 
    v["i71"] = 0.099979*np.tanh(((np.maximum(((np.where(data["BURO_DAYS_CREDIT_VAR"] < -99998, 0.318310, data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] ))), (((((data["BURO_DAYS_CREDIT_VAR"]) + (data["AMT_INCOME_TOTAL"]))/2.0))))) - ((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (np.tanh((data["AMT_INCOME_TOTAL"]))))*1.)))) 
    v["i72"] = 0.099800*np.tanh(np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"] < -99998, ((((np.tanh(((12.86315155029296875)))) - (0.318310))) / 2.0), ((np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"]>0, -2.0, ((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) * (data["NEW_EMPLOY_TO_BIRTH_RATIO"])) )) * 2.0) )) 
    v["i73"] = 0.100000*np.tanh((((((7.0)) * 2.0)) * (((((((((data["INSTAL_DPD_MEAN"]) * ((8.18208789825439453)))) + (data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]))) * ((6.44760274887084961)))) + (1.570796))))) 
    v["i74"] = 0.099530*np.tanh(((np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]))) + ((((np.where(data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]>0, (-1.0*((data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]))), data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] )) < ((((1.0) + ((((-1.0) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]))/2.0)))/2.0)))*1.)))) 
    v["i75"] = 0.099510*np.tanh(((np.minimum(((((1.570796) - (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["EXT_SOURCE_1"] ))))), ((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, data["NAME_INCOME_TYPE_Unemployed"], data["EXT_SOURCE_1"] )) - (data["NEW_SOURCES_PROD"])))))) * 2.0)) 
    v["i76"] = 0.098997*np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"] < -99998, ((-1.0) / 2.0), np.where(data["CLOSED_DAYS_CREDIT_MEAN"]>0, ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0), np.where(data["NEW_SOURCES_PROD"] < -99998, ((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0), ((2.0) * 2.0) ) ) )) 
    v["i77"] = 0.099680*np.tanh(np.tanh((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, np.where(data["REFUSED_APP_CREDIT_PERC_MAX"] < -99998, np.where(data["NEW_SCORES_STD"]<0, 1.0, data["NEW_SCORES_STD"] ), data["REFUSED_AMT_GOODS_PRICE_MAX"] ), (-1.0*((data["NEW_SCORES_STD"]))) )))) 
    v["i78"] = 0.099750*np.tanh(np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"] < -99998, 0.318310, (((-1.0*((((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) * 2.0)) * 2.0))))) - (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) + ((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (np.tanh((data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))*1.))))) )) 
    v["i79"] = 0.099996*np.tanh(((np.where(data["EXT_SOURCE_3"] < -99998, ((((data["ORGANIZATION_TYPE_Legal_Services"]) * 2.0)) * 2.0), (-1.0*(((((((((data["EXT_SOURCE_3"]) * 2.0)) + (1.570796))/2.0)) + (((data["EXT_SOURCE_3"]) * 2.0)))))) )) * 2.0)) 
    v["i80"] = 0.099060*np.tanh(np.where(((data["ACTIVE_DAYS_CREDIT_VAR"]) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) < -99998, 0.318310, ((((((((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))) * 2.0)) * 2.0)) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))) * 2.0)) * 2.0) )) 
    v["i81"] = 0.099949*np.tanh(np.where(data["REG_CITY_NOT_LIVE_CITY"]>0, (-1.0*((data["DAYS_BIRTH"]))), (((((data["REG_CITY_NOT_WORK_CITY"]) < (((np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((((data["DAYS_BIRTH"]) - (data["DAYS_REGISTRATION"])))))) - (data["DAYS_REGISTRATION"]))))*1.)) / 2.0) )) 
    v["i82"] = 0.099796*np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, ((-1.0) / 2.0), ((((((np.minimum(((data["NEW_SOURCES_PROD"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"])))) * (data["EXT_SOURCE_3"]))) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (np.tanh((-1.0)))) )) 
    v["i83"] = 0.099958*np.tanh(np.where(((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0)>0, -1.0, np.where(np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])))<0, (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.), (-1.0*((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * 2.0)))) ) )) 
    v["i84"] = 0.099960*np.tanh(np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]<0, ((np.tanh((np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, 1.0, ((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) + (np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]>0, (9.0), -2.0 ))) )))) / 2.0), (8.0) )) 
    v["i85"] = 0.099980*np.tanh(((((((np.tanh((-1.0))) - (data["NEW_SOURCES_PROD"]))) - (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))) * (((-2.0) - (((np.where(data["EXT_SOURCE_3"] < -99998, -1.0, data["EXT_SOURCE_3"] )) * 2.0)))))) 
    v["i86"] = 0.099797*np.tanh(np.where(data["EXT_SOURCE_1"] < -99998, ((np.where(data["ACTIVE_DAYS_CREDIT_VAR"]<0, data["DAYS_BIRTH"], data["ACTIVE_DAYS_CREDIT_VAR"] )) * 2.0), (-1.0*((((((((data["EXT_SOURCE_1"]) + (data["DAYS_BIRTH"]))) * 2.0)) + (data["DAYS_BIRTH"]))))) )) 
    v["i87"] = 0.099960*np.tanh(np.where(((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) + (((0.0) - (data["DAYS_BIRTH"])))) < -99998, data["DAYS_BIRTH"], np.maximum(((data["ORGANIZATION_TYPE_XNA"])), ((((((-1.0) + (data["DAYS_BIRTH"]))) * (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))) )) 
    v["i88"] = 0.099450*np.tanh((-1.0*(((((np.where(data["REFUSED_DAYS_DECISION_MEAN"]<0, ((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0), data["REFUSED_DAYS_DECISION_MEAN"] )) > (np.tanh(((((data["REFUSED_AMT_CREDIT_MEAN"]) > ((-1.0*((data["REFUSED_DAYS_DECISION_MEAN"])))))*1.)))))*1.))))) 
    v["i89"] = 0.099620*np.tanh(np.where(data["POS_SK_DPD_DEF_MAX"]>0, (-1.0*((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))), np.where(data["EXT_SOURCE_3"] < -99998, np.minimum(((data["POS_SK_DPD_DEF_MAX"])), ((-1.0))), np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, (-1.0*((data["EXT_SOURCE_3"]))), -1.0 ) ) )) 
    v["i90"] = 0.099901*np.tanh(np.tanh(((((-1.0*((np.maximum(((data["ACTIVE_DAYS_CREDIT_VAR"])), ((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))))))))) + (np.tanh((((((data["ACTIVE_DAYS_CREDIT_VAR"]) + (3.0))) * (data["ACTIVE_AMT_CREDIT_SUM_SUM"]))))))))) 
    v["i91"] = 0.096800*np.tanh((((np.where(data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]>0, np.tanh((-1.0)), np.tanh((data["CLOSED_DAYS_CREDIT_MEAN"])) )) + (np.minimum((((((data["CLOSED_DAYS_CREDIT_MEAN"]) < (0.318310))*1.))), (((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) < (data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"]))*1.))))))/2.0)) 
    v["i92"] = 0.098000*np.tanh(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]<0, (((((((1.570796) / 2.0)) * (data["DAYS_BIRTH"]))) < (((3.141593) * (data["OCCUPATION_TYPE_Core_staff"]))))*1.), np.minimum((((-1.0*((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]))))), ((data["DAYS_BIRTH"]))) )) 
    v["i93"] = 0.098996*np.tanh(((np.where(np.where(data["EXT_SOURCE_3"] < -99998, ((-1.0) + (data["NEW_SCORES_STD"])), (-1.0*((data["ORGANIZATION_TYPE_Bank"]))) )>0, data["EXT_SOURCE_3"], data["ORGANIZATION_TYPE_Bank"] )) * (((-1.0) + (data["NEW_SCORES_STD"]))))) 
    v["i94"] = 0.099400*np.tanh(np.tanh((np.maximum(((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]<0, ((data["NEW_SCORES_STD"]) - (2.0)), data["NEW_SCORES_STD"] ))), ((((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]<0, data["NAME_INCOME_TYPE_State_servant"], data["CLOSED_DAYS_CREDIT_VAR"] )) - (data["NEW_SCORES_STD"])))))))) 
    v["i95"] = 0.084695*np.tanh(np.where(data["REGION_RATING_CLIENT_W_CITY"]>0, data["ORGANIZATION_TYPE_Self_employed"], np.maximum(((np.where((((data["WALLSMATERIAL_MODE_Stone__brick"]) + (data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]))/2.0)>0, data["ORGANIZATION_TYPE_Self_employed"], data["REGION_RATING_CLIENT_W_CITY"] ))), (((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) + (data["WALLSMATERIAL_MODE_Stone__brick"]))/2.0)) / 2.0)))) )) 
    v["i96"] = 0.096899*np.tanh(np.where(data["DAYS_BIRTH"]>0, (((-1.0*(((((3.0) < (((data["DAYS_BIRTH"]) * 2.0)))*1.))))) * ((4.0))), np.where(data["REFUSED_DAYS_DECISION_MEAN"]<0, 0.318310, ((data["DAYS_BIRTH"]) * 2.0) ) )) 
    v["i97"] = 0.097551*np.tanh(((np.where(data["NEW_SOURCES_PROD"]>0, data["NAME_CONTRACT_TYPE_Cash_loans"], (((((-2.0) + ((((((data["NAME_INCOME_TYPE_Unemployed"]) > (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)) * 2.0)))) + ((((data["NAME_CONTRACT_TYPE_Cash_loans"]) < (data["ORGANIZATION_TYPE_Legal_Services"]))*1.)))/2.0) )) * 2.0)) 
    v["i98"] = 0.097900*np.tanh((((((data["ORGANIZATION_TYPE_Legal_Services"]) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((((np.maximum(((data["NAME_HOUSING_TYPE_Municipal_apartment"])), ((data["NAME_INCOME_TYPE_Unemployed"])))) - (((data["ORGANIZATION_TYPE_Industry__type_12"]) - (data["ORGANIZATION_TYPE_Business_Entity_Type_2"]))))) / 2.0)))/2.0)))/2.0)) - (data["ORGANIZATION_TYPE_Trade__type_6"]))) 
    v["i99"] = 0.059000*np.tanh(np.maximum(((np.tanh((np.tanh((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])))))), ((((data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"]) - ((-1.0*(((((np.maximum(((np.tanh((data["POS_SK_DPD_MEAN"])))), ((data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"])))) < (((data["POS_SK_DPD_MEAN"]) / 2.0)))*1.)))))))))) 
    v["i100"] = 0.096493*np.tanh(((((((data["REGION_POPULATION_RELATIVE"]) * (((((((np.tanh((data["REGION_POPULATION_RELATIVE"]))) < (0.636620))*1.)) + (np.tanh((data["NEW_CREDIT_TO_INCOME_RATIO"]))))/2.0)))) - (data["FONDKAPREMONT_MODE_reg_oper_spec_account"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) 
    v["i101"] = 0.074296*np.tanh(np.where((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)) + (data["NAME_INCOME_TYPE_Unemployed"]))) + (-2.0))) + (data["ORGANIZATION_TYPE_Realtor"]))/2.0)>0, 3.141593, ((data["OCCUPATION_TYPE_Secretaries"]) - (data["NAME_EDUCATION_TYPE_Academic_degree"])) )) 
    v["i102"] = 0.081810*np.tanh((((((((data["WALLSMATERIAL_MODE_Others"]) * 2.0)) + ((((((np.maximum(((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (data["NAME_TYPE_SUITE_Children"])))), ((data["ORGANIZATION_TYPE_Industry__type_1"])))) - (data["ORGANIZATION_TYPE_Industry__type_5"]))) + (data["ORGANIZATION_TYPE_Restaurant"]))/2.0)))/2.0)) - (data["ORGANIZATION_TYPE_Transport__type_1"]))) 
    v["i103"] = 0.098600*np.tanh(np.where(data["DEF_30_CNT_SOCIAL_CIRCLE"] < -99998, data["DEF_30_CNT_SOCIAL_CIRCLE"], ((data["DEF_30_CNT_SOCIAL_CIRCLE"]) - (((np.where(data["DEF_30_CNT_SOCIAL_CIRCLE"]>0, -2.0, ((data["ORGANIZATION_TYPE_Agriculture"]) - (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])) )) * (data["DEF_30_CNT_SOCIAL_CIRCLE"])))) )) 
    v["i104"] = 0.091970*np.tanh(((((((data["FLAG_DOCUMENT_3"]) * (np.maximum(((data["AMT_INCOME_TOTAL"])), (((-1.0*(((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > ((((data["AMT_INCOME_TOTAL"]) > (data["FLAG_DOCUMENT_3"]))*1.)))*1.)))))))))) - (data["ORGANIZATION_TYPE_Hotel"]))) - (data["ORGANIZATION_TYPE_Hotel"]))) 
    v["i105"] = 0.099940*np.tanh(((np.where(data["EXT_SOURCE_2"]<0, data["FONDKAPREMONT_MODE_reg_oper_spec_account"], ((((((((data["FONDKAPREMONT_MODE_reg_oper_spec_account"]) + (1.0))/2.0)) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)) - (((data["EXT_SOURCE_2"]) * 2.0))) )) - (((data["EXT_SOURCE_2"]) * 2.0)))) 
    v["i106"] = 0.096000*np.tanh(np.where((((data["NEW_INC_PER_CHLD"]) + ((((data["REGION_POPULATION_RELATIVE"]) + (data["NEW_INC_PER_CHLD"]))/2.0)))/2.0)<0, ((np.where(data["CLOSED_DAYS_CREDIT_VAR"] < -99998, data["NAME_EDUCATION_TYPE_Higher_education"], data["REGION_POPULATION_RELATIVE"] )) / 2.0), (((data["DAYS_REGISTRATION"]) < (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.) )) 
    v["i107"] = 0.092699*np.tanh((((((data["REGION_RATING_CLIENT"]) < ((((((data["NEW_SOURCES_PROD"]) < (-1.0))*1.)) - ((-1.0*((((np.minimum(((((data["REGION_RATING_CLIENT"]) * (data["NEW_SOURCES_PROD"])))), ((data["NEW_SOURCES_PROD"])))) / 2.0))))))))*1.)) * 2.0)) 
    v["i108"] = 0.099960*np.tanh(np.minimum(((((((data["DAYS_BIRTH"]) * (((data["REGION_RATING_CLIENT_W_CITY"]) - (0.318310))))) - (np.where(np.minimum(((data["DAYS_BIRTH"])), ((data["REGION_RATING_CLIENT_W_CITY"])))<0, data["REGION_RATING_CLIENT_W_CITY"], 0.636620 ))))), (((1.00496315956115723))))) 
    v["i109"] = 0.089999*np.tanh((((data["ORGANIZATION_TYPE_Mobile"]) + ((((np.minimum((((-1.0*((data["ORGANIZATION_TYPE_Industry__type_9"]))))), ((data["ORGANIZATION_TYPE_Other"])))) + ((((((data["ORGANIZATION_TYPE_Other"]) / 2.0)) + (np.minimum((((-1.0*((data["ORGANIZATION_TYPE_Industry__type_9"]))))), ((data["FLAG_EMP_PHONE"])))))/2.0)))/2.0)))/2.0)) 
    v["i110"] = 0.099760*np.tanh(np.where((((0.318310) < (((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)))*1.)>0, (((data["REGION_POPULATION_RELATIVE"]) < (data["AMT_ANNUITY"]))*1.), ((-1.0) + (((data["AMT_ANNUITY"]) * (data["AMT_ANNUITY"])))) )) 
    v["i111"] = 0.100000*np.tanh((-1.0*((np.where((-1.0*(((((data["AMT_CREDIT"]) > ((((3.141593) + (0.636620))/2.0)))*1.))))<0, (6.10089683532714844), ((((np.maximum(((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])), ((data["AMT_CREDIT"])))) / 2.0)) / 2.0) ))))) 
    v["i112"] = 0.094904*np.tanh(np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NEW_CREDIT_TO_INCOME_RATIO"]))<0, ((-1.0) - (np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["NAME_EDUCATION_TYPE_Academic_degree"] ))), ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * 2.0)) + (1.0)) )) 
    v["i113"] = 0.099204*np.tanh((((((np.minimum(((data["AMT_CREDIT"])), (((((1.63399493694305420)) / 2.0))))) > (((np.maximum(((data["BURO_DAYS_CREDIT_VAR"])), ((1.570796)))) / 2.0)))*1.)) * (((data["NEW_CREDIT_TO_INCOME_RATIO"]) * 2.0)))) 
    v["i114"] = 0.004658*np.tanh(np.where(data["REG_CITY_NOT_WORK_CITY"]>0, (-1.0*(((((data["BURO_DAYS_CREDIT_VAR"]) > (data["NAME_EDUCATION_TYPE_Academic_degree"]))*1.)))), ((((-1.0*((data["NAME_EDUCATION_TYPE_Academic_degree"])))) + ((((data["ORGANIZATION_TYPE_Legal_Services"]) + ((((data["BURO_DAYS_CREDIT_VAR"]) > (data["REG_CITY_NOT_WORK_CITY"]))*1.)))/2.0)))/2.0) )) 
    v["i115"] = 0.090201*np.tanh(((((((((((np.where(data["CLOSED_DAYS_CREDIT_VAR"] < -99998, data["ORGANIZATION_TYPE_Transport__type_1"], np.where(data["CLOSED_DAYS_CREDIT_VAR"]<0, data["CLOSED_DAYS_CREDIT_VAR"], data["REG_CITY_NOT_WORK_CITY"] ) )) * (data["NAME_EDUCATION_TYPE_Academic_degree"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i116"] = 0.083965*np.tanh(np.where(data["BURO_MONTHS_BALANCE_MAX_MAX"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], np.where(data["BURO_STATUS_2_MEAN_MEAN"]>0, ((((data["BURO_MONTHS_BALANCE_MIN_MIN"]) * 2.0)) + (data["BURO_STATUS_4_MEAN_MEAN"])), ((np.minimum(((0.318310)), ((data["BURO_MONTHS_BALANCE_MIN_MIN"])))) - (data["BURO_MONTHS_BALANCE_MIN_MIN"])) ) )) 
    v["i117"] = 0.096950*np.tanh(np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((10.0)))) < (((((((10.0)) > ((7.85062408447265625)))*1.)) / 2.0)))*1.))), (((((10.0)) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((8.0))))))))) 
    v["i118"] = 0.099808*np.tanh(np.minimum(((np.minimum(((data["AMT_INCOME_TOTAL"])), (((-1.0*((np.minimum(((data["NAME_CONTRACT_TYPE_Cash_loans"])), ((data["AMT_INCOME_TOTAL"]))))))))))), (((((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (data["NAME_CONTRACT_TYPE_Cash_loans"]))) * (data["AMT_INCOME_TOTAL"]))) + (data["NAME_CONTRACT_TYPE_Cash_loans"]))/2.0))))) 
    v["i119"] = 0.093920*np.tanh(((np.minimum(((data["EXT_SOURCE_1"])), ((np.where(np.where(data["ORGANIZATION_TYPE_Legal_Services"]>0, 0.636620, data["EXT_SOURCE_1"] )>0, 0.636620, (-1.0*((1.570796))) ))))) - (data["EXT_SOURCE_1"]))) 
    v["i120"] = 0.090007*np.tanh((((0.318310) + (((((data["ORGANIZATION_TYPE_Legal_Services"]) + ((((-1.0*((np.maximum(((((data["NAME_INCOME_TYPE_Student"]) - (data["ORGANIZATION_TYPE_Industry__type_12"])))), ((data["ORGANIZATION_TYPE_Industry__type_12"]))))))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))))) + (data["NAME_INCOME_TYPE_Unemployed"]))))/2.0)) 
    v["i121"] = 0.096997*np.tanh(np.minimum((((((data["EXT_SOURCE_1"]) < (1.0))*1.))), ((np.where(data["EXT_SOURCE_1"]>0, data["NEW_INC_PER_CHLD"], (((1.570796) < (np.where(data["EXT_SOURCE_1"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (-1.0*((data["EXT_SOURCE_1"]))) )))*1.) ))))) 
    v["i122"] = 0.094855*np.tanh(((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * (data["CODE_GENDER"]))) + ((((0.636620) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.)))) 
    v["i123"] = 0.100000*np.tanh(((((np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], (((((-1.0) - (data["EXT_SOURCE_3"]))) > (0.636620))*1.) )) * 2.0)) + ((-1.0*(((((1.0) < (data["EXT_SOURCE_3"]))*1.))))))) 
    v["i124"] = 0.097490*np.tanh((-1.0*((np.where((-1.0*(((((data["REFUSED_AMT_GOODS_PRICE_MEAN"]) > ((-1.0*(((((-1.0) > (((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0)))*1.))))))*1.))))<0, data["REFUSED_DAYS_DECISION_MAX"], (0.12027385830879211) ))))) 
    v["i125"] = 0.091002*np.tanh((-1.0*((np.tanh(((((np.where(np.where(data["REFUSED_AMT_CREDIT_MEAN"]>0, data["REFUSED_AMT_CREDIT_MAX"], data["REFUSED_AMT_APPLICATION_MEAN"] )>0, data["REFUSED_CNT_PAYMENT_SUM"], np.where(data["REFUSED_CNT_PAYMENT_SUM"]>0, data["REFUSED_AMT_APPLICATION_MAX"], data["REFUSED_AMT_CREDIT_MEAN"] ) )) > (data["REFUSED_AMT_CREDIT_MAX"]))*1.))))))) 
    v["i126"] = 0.091349*np.tanh(np.where(((data["DAYS_BIRTH"]) - (data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]))<0, -1.0, (((data["DAYS_BIRTH"]) < ((((((data["REFUSED_AMT_CREDIT_MAX"]) > (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]))*1.)) - (1.570796))))*1.) )) 
    v["i127"] = 0.099499*np.tanh((((data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"]) > (((np.tanh((np.where(data["CLOSED_AMT_ANNUITY_MAX"]>0, np.maximum((((-1.0*((data["NAME_EDUCATION_TYPE_Academic_degree"]))))), ((data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]))), data["CLOSED_AMT_ANNUITY_MEAN"] )))) - (data["CLOSED_AMT_ANNUITY_MAX"]))))*1.)) 
    v["i128"] = 0.009900*np.tanh(np.where(data["FLAG_EMAIL"]>0, ((data["AMT_INCOME_TOTAL"]) - ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (0.636620))/2.0))), ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.maximum(((data["NAME_TYPE_SUITE_Group_of_people"])), (((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (data["FLAG_EMAIL"]))/2.0)))))) )) 
    v["i129"] = 0.096249*np.tanh((((((data["ORGANIZATION_TYPE_Postal"]) + (((((((((((1.0)) + (data["ORGANIZATION_TYPE_Postal"]))) < (data["AMT_INCOME_TOTAL"]))*1.)) - (data["ORGANIZATION_TYPE_Trade__type_4"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))))/2.0)) + ((-1.0*((data["ORGANIZATION_TYPE_Transport__type_1"])))))) 
    v["i130"] = 0.098970*np.tanh((((((((((-1.0*((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((np.tanh((-1.0))) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))/2.0)) / 2.0)) < (data["ORGANIZATION_TYPE_Bank"]))*1.))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i131"] = 0.092860*np.tanh((((data["ORGANIZATION_TYPE_Telecom"]) + (np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, (-1.0*((data["NEW_CREDIT_TO_INCOME_RATIO"]))), np.where(data["FLAG_EMAIL"]<0, (((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.), (-1.0*((data["FLAG_EMAIL"]))) ) )))/2.0)) 
    v["i132"] = 0.099999*np.tanh((((((((np.maximum(((1.570796)), ((data["NEW_SCORES_STD"])))) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)) + (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, (((3.141593) < (data["NEW_SCORES_STD"]))*1.), np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"])) )))) * 2.0)) 
    v["i133"] = 0.099840*np.tanh((((((data["DAYS_BIRTH"]) * (data["NEW_SCORES_STD"]))) + (np.minimum(((((((data["DAYS_BIRTH"]) * 2.0)) * ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))/2.0)) 
    v["i134"] = 0.099021*np.tanh(((np.where(((((((3.0) - (1.570796))) - (data["NAME_INCOME_TYPE_Student"]))) - (data["DAYS_BIRTH"]))>0, ((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (((data["NEW_SCORES_STD"]) * 2.0))), -2.0 )) * 2.0)) 
    v["i135"] = 0.084999*np.tanh(np.where(data["NAME_HOUSING_TYPE_With_parents"]>0, (((data["AMT_ANNUITY"]) < (0.318310))*1.), ((1.0) - (np.maximum(((data["AMT_ANNUITY"])), ((np.maximum(((0.636620)), (((((data["AMT_ANNUITY"]) < (data["NAME_HOUSING_TYPE_With_parents"]))*1.))))))))) )) 
    v["i136"] = 0.095405*np.tanh(np.where((((((data["NAME_INCOME_TYPE_Student"]) + (np.minimum(((data["INSTAL_PAYMENT_PERC_VAR"])), ((data["INSTAL_PAYMENT_DIFF_STD"])))))) > (((data["INSTAL_PAYMENT_DIFF_VAR"]) * 2.0)))*1.)>0, 0.636620, np.minimum((((-1.0*((0.318310))))), ((data["INSTAL_AMT_PAYMENT_STD"]))) )) 
    v["i137"] = 0.099960*np.tanh(np.where((((((np.where(data["INSTAL_PAYMENT_PERC_MAX"]>0, (-1.0*((data["NAME_INCOME_TYPE_Maternity_leave"]))), data["NAME_EDUCATION_TYPE_Academic_degree"] )) / 2.0)) < (data["INSTAL_PAYMENT_PERC_MEAN"]))*1.)>0, -2.0, (((data["INSTAL_PAYMENT_PERC_MAX"]) > (data["INSTAL_PAYMENT_PERC_MEAN"]))*1.) )) 
    v["i138"] = 0.094701*np.tanh(np.where(data["YEARS_BEGINEXPLUATATION_MODE"] < -99998, (-1.0*((0.318310))), (-1.0*((np.where(data["YEARS_BEGINEXPLUATATION_MODE"]<0, data["LIVINGAREA_AVG"], ((data["YEARS_BEGINEXPLUATATION_MODE"]) + (((data["YEARS_BEGINEXPLUATATION_MODE"]) * 2.0))) )))) )) 
    v["i139"] = 0.099971*np.tanh(((((((np.where(data["ORGANIZATION_TYPE_Realtor"]<0, np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((((2.0) - (data["AMT_ANNUITY"]))))), 2.0 )) - (data["NAME_INCOME_TYPE_Student"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) * 2.0)) 
    v["i140"] = 0.099589*np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (0.318310))*1.)) - ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) > (((data["AMT_INCOME_TOTAL"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))))*1.)))) 
    v["i141"] = 0.061000*np.tanh(((((-1.0*(((((data["OCCUPATION_TYPE_Drivers"]) < (data["REFUSED_AMT_ANNUITY_MEAN"]))*1.))))) + (np.where(data["REFUSED_CNT_PAYMENT_MEAN"] < -99998, np.tanh((data["NAME_EDUCATION_TYPE_Higher_education"])), np.minimum((((-1.0*((data["NAME_EDUCATION_TYPE_Higher_education"]))))), ((data["REFUSED_CNT_PAYMENT_MEAN"]))) )))/2.0)) 
    v["i142"] = 0.029650*np.tanh(((np.minimum(((((((((data["NAME_INCOME_TYPE_Student"]) * (data["NAME_INCOME_TYPE_Student"]))) + (data["AMT_INCOME_TOTAL"]))) * (data["NAME_TYPE_SUITE_Group_of_people"])))), ((((data["NAME_TYPE_SUITE_Group_of_people"]) * (((data["NAME_INCOME_TYPE_Working"]) + (data["OCCUPATION_TYPE_Drivers"])))))))) * 2.0)) 
    v["i143"] = 0.096011*np.tanh(np.minimum(((((0.318310) * (0.318310)))), ((((np.minimum(((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0))), ((0.636620)))) * (np.where(data["NAME_INCOME_TYPE_Maternity_leave"]<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_ANNUITY"] ))))))) 
    v["i144"] = 0.074027*np.tanh(((np.tanh(((((data["DAYS_EMPLOYED"]) < ((((-1.0*((np.maximum(((data["NEW_EMPLOY_TO_BIRTH_RATIO"])), ((((data["NAME_INCOME_TYPE_Pensioner"]) - ((-1.0*((0.318310)))))))))))) / 2.0)))*1.)))) / 2.0)) 
    v["i145"] = 0.098999*np.tanh(((((((((((1.570796) - ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (0.318310))*1.)) / 2.0)) > (((1.570796) - ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))*1.)) 
    v["i146"] = 0.064730*np.tanh((-1.0*(((((((data["BURO_MONTHS_BALANCE_MAX_MAX"]) > (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.)) * ((((((((0.0) > (data["BURO_STATUS_0_MEAN_MEAN"]))*1.)) + ((-1.0*((data["BURO_MONTHS_BALANCE_MAX_MAX"])))))) + (data["BURO_STATUS_0_MEAN_MEAN"])))))))) 
    v["i147"] = 0.099995*np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((((0.13536217808723450)) / 2.0)))*1.)) * (((((((((0.636620) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)) > ((0.13536217808723450)))*1.)) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))))) 
    v["i148"] = 0.099000*np.tanh(np.minimum((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) < (data["EXT_SOURCE_3"]))*1.))), ((((((np.where(data["NEW_SOURCES_PROD"]>0, data["NEW_DOC_IND_KURT"], (((((1.0) + (data["NEW_DOC_IND_KURT"]))/2.0)) - (data["EXT_SOURCE_3"])) )) * 2.0)) * 2.0))))) 
    v["i149"] = 0.099300*np.tanh(np.minimum((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (data["ORGANIZATION_TYPE_Industry__type_2"]))*1.))), ((((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), (((-1.0*((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))) - (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * 2.0) ))))))) 
    v["i150"] = 0.020000*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["AMT_INCOME_TOTAL"], np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.0, np.where(np.maximum(((data["ACTIVE_DAYS_CREDIT_VAR"])), ((data["AMT_INCOME_TOTAL"])))>0, data["ACTIVE_DAYS_CREDIT_VAR"], (-1.0*((((data["NAME_INCOME_TYPE_Working"]) / 2.0)))) ) ) )) 
    v["i151"] = 0.099731*np.tanh(np.where((((((data["NAME_EDUCATION_TYPE_Lower_secondary"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))<0, (((((((data["NAME_EDUCATION_TYPE_Lower_secondary"]) / 2.0)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) + (data["NAME_EDUCATION_TYPE_Lower_secondary"])), ((data["NAME_EDUCATION_TYPE_Lower_secondary"]) * 2.0) )) 
    v["i152"] = 0.034897*np.tanh(np.where(data["NEW_DOC_IND_KURT"]>0, ((np.where((-1.0*((data["EXT_SOURCE_3"])))>0, (-1.0*((0.636620))), data["AMT_CREDIT"] )) / 2.0), np.where(data["NAME_INCOME_TYPE_Student"]<0, (-1.0*((data["EXT_SOURCE_3"]))), data["NEW_DOC_IND_KURT"] ) )) 
    v["i153"] = 0.088029*np.tanh(((((((1.570796) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((1.570796) + (-1.0))))*1.)))*1.)) 
    v["i154"] = 0.097020*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * ((((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (data["AMT_INCOME_TOTAL"]))*1.)) - (((((((data["AMT_ANNUITY"]) < (0.318310))*1.)) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.)))))) 
    v["i155"] = 0.060999*np.tanh((-1.0*(((((((((((((data["NAME_INCOME_TYPE_Student"]) - (3.0))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (3.141593))) > (1.0))*1.)) * 2.0))))) 
    v["i156"] = 0.099598*np.tanh(((((data["AMT_INCOME_TOTAL"]) - ((((np.tanh((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.)))) * ((((np.tanh(((-1.0*((data["AMT_INCOME_TOTAL"])))))) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.)))) 
    v["i157"] = 0.099740*np.tanh((((((((3.0) < (((((np.where(data["AMT_CREDIT"]<0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((-1.0*((0.636620)))))*1.), 0.636620 )) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * 2.0)))*1.)) * 2.0)) * 2.0)) 
    v["i158"] = 0.098100*np.tanh(((((np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((np.where(data["NAME_INCOME_TYPE_Working"]>0, (-1.0*((((data["NEW_DOC_IND_KURT"]) * (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))))), data["NEW_DOC_IND_KURT"] )) - (data["NAME_INCOME_TYPE_Working"])))))) * (data["FLAG_EMP_PHONE"]))) / 2.0)) 
    v["i159"] = 0.099450*np.tanh(((((data["NEW_DOC_IND_KURT"]) / 2.0)) * (np.where(((0.318310) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))>0, np.where(data["EXT_SOURCE_3"]<0, ((data["AMT_ANNUITY"]) / 2.0), data["EXT_SOURCE_3"] ), data["EXT_SOURCE_3"] )))) 
    v["i160"] = 0.099801*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"]>0, np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]<0, 0.0, (((4.0)) - (data["ACTIVE_DAYS_CREDIT_VAR"])) ), (((data["ACTIVE_DAYS_CREDIT_VAR"]) > (((1.0) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))*1.) )) 
    v["i161"] = 0.076150*np.tanh((((((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["ACTIVE_DAYS_CREDIT_VAR"] )))*1.)) / 2.0)) - ((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) > (np.where(data["ACTIVE_DAYS_CREDIT_VAR"]<0, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], data["NAME_INCOME_TYPE_Maternity_leave"] )))*1.)))) 
    v["i162"] = 0.099900*np.tanh(((((data["EXT_SOURCE_2"]) * (((((np.tanh((np.tanh((np.minimum(((((data["EXT_SOURCE_2"]) * 2.0))), ((data["AMT_ANNUITY"])))))))) - (data["EXT_SOURCE_2"]))) * 2.0)))) * (((data["EXT_SOURCE_2"]) * 2.0)))) 
    v["i163"] = 0.081001*np.tanh((((np.minimum(((data["OCCUPATION_TYPE_High_skill_tech_staff"])), ((data["HOUR_APPR_PROCESS_START"])))) > (np.where(data["AMT_INCOME_TOTAL"]<0, np.minimum(((data["REGION_RATING_CLIENT_W_CITY"])), ((np.where(data["REGION_RATING_CLIENT"]<0, ((data["AMT_INCOME_TOTAL"]) / 2.0), data["NAME_INCOME_TYPE_Maternity_leave"] )))), data["AMT_INCOME_TOTAL"] )))*1.)) 
    v["i164"] = 0.099752*np.tanh(((np.where(data["CLOSED_DAYS_CREDIT_VAR"]<0, np.minimum(((np.where(data["AMT_INCOME_TOTAL"]<0, ((data["REGION_RATING_CLIENT_W_CITY"]) - (data["WALLSMATERIAL_MODE_Panel"])), (-1.0*((data["REGION_RATING_CLIENT_W_CITY"]))) ))), ((data["NAME_INCOME_TYPE_Maternity_leave"]))), data["WALLSMATERIAL_MODE_Panel"] )) * (data["REGION_RATING_CLIENT_W_CITY"]))) 
    v["i165"] = 0.098940*np.tanh(np.maximum(((data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"])), ((np.where(data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]<0, np.where(data["CLOSED_DAYS_CREDIT_VAR"]<0, np.tanh((data["AMT_INCOME_TOTAL"])), data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"] ), np.where(np.tanh((data["AMT_INCOME_TOTAL"]))<0, data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"], data["CLOSED_DAYS_CREDIT_VAR"] ) ))))) 
    v["i166"] = 0.083961*np.tanh((((3.141593) < ((((-1.0*((np.minimum(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["DAYS_BIRTH"], np.where(data["DAYS_BIRTH"]>0, data["DAYS_BIRTH"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] ) ))), ((data["ORGANIZATION_TYPE_Advertising"]))))))) * 2.0)))*1.)) 
    v["i167"] = 0.099970*np.tanh(((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) * (((data["NAME_FAMILY_STATUS_Married"]) * (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((data["ORGANIZATION_TYPE_Advertising"]) + (data["OCCUPATION_TYPE_Low_skill_Laborers"]))))) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)))))))) * 2.0)) 
    v["i168"] = 0.069500*np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (np.maximum(((np.tanh(((((4.77533054351806641)) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NAME_TYPE_SUITE_Group_of_people"])))))))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) - (data["OCCUPATION_TYPE_Low_skill_Laborers"]))) 
    v["i169"] = 0.098609*np.tanh(np.minimum(((np.maximum((((((data["ACTIVE_DAYS_CREDIT_VAR"]) < ((((((-2.0) / 2.0)) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))*1.))), ((data["AMT_ANNUITY"]))))), ((((((((0.318310) < (data["AMT_ANNUITY"]))*1.)) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.))))) 
    v["i170"] = 0.099983*np.tanh((((((-1.0*((np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, 3.0, np.where(data["ORGANIZATION_TYPE_Advertising"]<0, data["ORGANIZATION_TYPE_Industry__type_2"], ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["ACTIVE_DAYS_CREDIT_VAR"]))) - (3.0)) ) ))))) * 2.0)) * 2.0)) 
    v["i171"] = 0.096960*np.tanh(np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))>0, data["OCCUPATION_TYPE_Accountants"], (((((((data["NAME_EDUCATION_TYPE_Higher_education"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((((data["OCCUPATION_TYPE_Accountants"]) < (data["NAME_TYPE_SUITE_Other_A"]))*1.)))*1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) )) 
    v["i172"] = 0.099790*np.tanh(((((np.where(data["NAME_TYPE_SUITE_Other_A"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((((-1.0*((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))))) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, 3.0, ((data["NAME_TYPE_SUITE_Other_A"]) * 2.0) )))*1.) )) * 2.0)) * 2.0)) 
    v["i173"] = 0.093300*np.tanh(np.where(((data["DAYS_BIRTH"]) / 2.0)<0, (((data["NAME_TYPE_SUITE_Other_A"]) < (((data["DAYS_BIRTH"]) / 2.0)))*1.), (((data["NAME_TYPE_SUITE_Other_A"]) > ((((5.23197555541992188)) - (data["ACTIVE_DAYS_CREDIT_VAR"]))))*1.) )) 
    v["i174"] = 0.085000*np.tanh((-1.0*((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["NAME_TYPE_SUITE_Other_A"], (((np.where(data["ACTIVE_DAYS_CREDIT_VAR"]<0, (-1.0*((0.318310))), data["NAME_TYPE_SUITE_Other_A"] )) < (data["REFUSED_AMT_ANNUITY_MEAN"]))*1.) ))))) 
    v["i175"] = 0.099900*np.tanh((((((data["ACTIVE_AMT_ANNUITY_MAX"]) > (np.where(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"] < -99998, (6.76153230667114258), (((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]))/2.0) )<0, ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) / 2.0)) / 2.0), 0.318310 )))*1.)) * 2.0)) 
    v["i176"] = 0.095999*np.tanh((-1.0*((np.maximum(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["NAME_EDUCATION_TYPE_Lower_secondary"], 0.318310 ))), ((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, ((((3.92553544044494629)) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))*1.), ((0.318310) / 2.0) )))))))) 
    v["i177"] = 0.040399*np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((np.tanh((1.0))) < (((data["NEW_CREDIT_TO_INCOME_RATIO"]) + ((((-1.0) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))))*1.) )) / 2.0)) 
    v["i178"] = 0.094910*np.tanh(np.where(data["DAYS_BIRTH"]<0, (((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (-1.0))*1.), np.minimum(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (1.0)))), (((((data["DAYS_BIRTH"]) < (((-1.0) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))))*1.)))) )) 
    v["i179"] = 0.099100*np.tanh(np.minimum(((np.where(data["LIVE_REGION_NOT_WORK_REGION"]<0, (((data["OCCUPATION_TYPE_Sales_staff"]) > (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.), data["OCCUPATION_TYPE_Sales_staff"] ))), ((((((np.minimum(((((data["OCCUPATION_TYPE_Sales_staff"]) * 2.0))), ((data["ACTIVE_DAYS_CREDIT_VAR"])))) / 2.0)) * (data["NAME_INCOME_TYPE_Student"])))))) 
    v["i180"] = 0.095002*np.tanh(((np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)))/2.0)<0, ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * (data["NEW_CREDIT_TO_INCOME_RATIO"])), data["NEW_CREDIT_TO_INCOME_RATIO"] ))))) * 2.0)) 
    v["i181"] = 0.068969*np.tanh((((((((data["NAME_CONTRACT_TYPE_Cash_loans"]) + (((((data["DAYS_BIRTH"]) * ((((-1.0*((data["DAYS_BIRTH"])))) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))))) * (data["NAME_CONTRACT_TYPE_Revolving_loans"]))))/2.0)) + (data["ORGANIZATION_TYPE_Mobile"]))) + (data["ORGANIZATION_TYPE_Mobile"]))) 
    v["i182"] = 0.065961*np.tanh(np.minimum(((np.where(data["CLOSED_DAYS_CREDIT_VAR"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.where(data["OCCUPATION_TYPE_Sales_staff"]>0, data["CLOSED_DAYS_CREDIT_VAR"], ((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (((data["OCCUPATION_TYPE_Sales_staff"]) * 2.0))) ) ))), (((((0.318310) > (data["CLOSED_DAYS_CREDIT_VAR"]))*1.))))) 
    v["i183"] = 0.096998*np.tanh(np.maximum((((((6.0)) * (((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (((data["AMT_CREDIT"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))))), ((np.minimum(((((data["AMT_CREDIT"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((data["OCCUPATION_TYPE_Sales_staff"]))))))) 
    v["i184"] = 0.099830*np.tanh((((((((np.maximum((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))/2.0))), ((((np.tanh((1.570796))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) + (-2.0))) > (data["ORGANIZATION_TYPE_Trade__type_4"]))*1.)) / 2.0)) 
    v["i185"] = 0.094980*np.tanh((((((data["AMT_INCOME_TOTAL"]) > (np.maximum(((np.maximum(((((data["DAYS_BIRTH"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((data["DAYS_BIRTH"]))))), (((((data["CLOSED_DAYS_CREDIT_VAR"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.))))))*1.)) * 2.0)) 
    v["i186"] = 0.097300*np.tanh(np.minimum(((((data["AMT_ANNUITY"]) * ((((((data["AMT_ANNUITY"]) * ((-1.0*((data["NEW_DOC_IND_KURT"])))))) + (data["DAYS_BIRTH"]))/2.0))))), ((data["ORGANIZATION_TYPE_Trade__type_4"])))) 
    v["i187"] = 0.098399*np.tanh((-1.0*((np.tanh(((((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["NAME_HOUSING_TYPE_Office_apartment"])))) + ((((((((data["NAME_HOUSING_TYPE_Office_apartment"]) > (((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.)) * 2.0)) * 2.0)))/2.0))))))) 
    v["i188"] = 0.037011*np.tanh(((data["ORGANIZATION_TYPE_Trade__type_6"]) * (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, ((((np.where(data["AMT_ANNUITY"]<0, data["AMT_ANNUITY"], 1.570796 )) * 2.0)) * 2.0), ((1.0) - (data["AMT_ANNUITY"])) )))) 
    v["i189"] = 0.075050*np.tanh((((3.141593) < ((-1.0*(((((-1.0*(((((-1.0) < (np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), (((-1.0*((data["AMT_ANNUITY"]))))))))*1.))))) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))))))))*1.)) 
    v["i190"] = 0.097999*np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["AMT_ANNUITY"]))*1.)) * (data["WALLSMATERIAL_MODE_Panel"]))) * (np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) - (data["AMT_INCOME_TOTAL"]))) - (data["AMT_INCOME_TOTAL"]))))))) 
    v["i191"] = 0.070000*np.tanh(((((np.where(data["NONLIVINGAREA_MODE"]>0, (((((2.0) < (data["FLOORSMAX_MODE"]))*1.)) - ((((data["NONLIVINGAREA_MODE"]) < (data["NONLIVINGAREA_AVG"]))*1.))), ((data["NONLIVINGAREA_AVG"]) - (data["NONLIVINGAREA_MODE"])) )) * 2.0)) * 2.0)) 
    v["i192"] = 0.092500*np.tanh(((((((((-1.0) + (data["FLOORSMAX_MEDI"]))/2.0)) > ((0.13370278477668762)))*1.)) - ((((data["FLOORSMAX_MODE"]) > ((((((data["FLOORSMAX_MODE"]) * (data["YEARS_BUILD_MEDI"]))) < (np.tanh((data["FLOORSMAX_MEDI"]))))*1.)))*1.)))) 
    v["i193"] = 0.095001*np.tanh(((((((((((data["NONLIVINGAREA_MODE"]) + ((((data["YEARS_BUILD_MODE"]) < ((-1.0*((data["NONLIVINGAREA_MODE"])))))*1.)))) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)) > (0.636620))*1.)) - ((((0.318310) < (data["NONLIVINGAREA_MODE"]))*1.)))) 
    v["i194"] = 0.095010*np.tanh((((((data["FLOORSMAX_MODE"]) * (((1.570796) * (data["APARTMENTS_MEDI"]))))) < (np.where(data["NONLIVINGAREA_MEDI"] < -99998, np.where(data["FLOORSMAX_MEDI"]>0, 0.318310, data["APARTMENTS_AVG"] ), -1.0 )))*1.)) 
    v["i195"] = 0.079990*np.tanh((-1.0*((((((((data["APARTMENTS_AVG"]) < (np.where(data["FLOORSMAX_AVG"]<0, np.tanh((data["ENTRANCES_MODE"])), np.where(data["NONLIVINGAREA_MEDI"] < -99998, data["APARTMENTS_AVG"], 0.318310 ) )))*1.)) + (np.tanh((data["ENTRANCES_AVG"]))))/2.0))))) 
    v["i196"] = 0.091496*np.tanh(((((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * ((((2.0) > ((((((3.141593) + (np.where(data["AMT_ANNUITY"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_ANNUITY"] )))/2.0)) * 2.0)))*1.)))) * 2.0)) 
    v["i197"] = 0.074650*np.tanh(((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), (((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (np.minimum((((-1.0*((((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (data["ORGANIZATION_TYPE_Trade__type_4"]))))))), ((data["NAME_CONTRACT_TYPE_Revolving_loans"])))))) < (data["AMT_INCOME_TOTAL"]))*1.))))))) 
    v["i198"] = 0.096000*np.tanh(((((-1.0*((np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NAME_TYPE_SUITE_Other_A"]))<0, ((data["NAME_TYPE_SUITE_Other_A"]) / 2.0), data["NAME_INCOME_TYPE_Student"] ))))) > ((((0.636620) + ((((0.318310) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)))/2.0)))*1.)) 
    v["i199"] = 0.022986*np.tanh(((((((np.where(data["AMT_ANNUITY"]>0, 2.0, ((data["NAME_EDUCATION_TYPE_Academic_degree"]) - (2.0)) )) - (data["AMT_ANNUITY"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) * (data["ORGANIZATION_TYPE_Trade__type_6"]))) 
    v["i200"] = 0.059999*np.tanh(np.minimum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (1.570796))*1.)>0, (-1.0*((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["ORGANIZATION_TYPE_Trade__type_4"], ((data["AMT_ANNUITY"]) + (-2.0)) )))), data["AMT_ANNUITY"] ))))) 
    v["i201"] = 0.087580*np.tanh(((np.where(((data["BURO_STATUS_2_MEAN_MEAN"]) + (data["BURO_STATUS_4_MEAN_MEAN"]))<0, (((0.318310) < ((((data["AMT_ANNUITY"]) + (((data["BURO_STATUS_2_MEAN_MEAN"]) + (data["BURO_STATUS_4_MEAN_MEAN"]))))/2.0)))*1.), -2.0 )) * 2.0)) 
    v["i202"] = 0.099718*np.tanh((((((-1.0*(((((np.where(np.where(data["BURO_STATUS_0_MEAN_MEAN"]<0, data["BURO_STATUS_X_MEAN_MEAN"], data["BURO_STATUS_0_MEAN_MEAN"] )<0, ((np.tanh((-1.0))) / 2.0), (2.05887126922607422) )) < (data["BURO_STATUS_0_MEAN_MEAN"]))*1.))))) * 2.0)) * 2.0)) 
    v["i203"] = 0.097371*np.tanh((((((np.where(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))>0, 2.0, ((data["NAME_EDUCATION_TYPE_Academic_degree"]) / 2.0) )) < (((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))*1.)) * 2.0)) 
    v["i204"] = 0.094501*np.tanh(((np.where(((np.where(data["BURO_MONTHS_BALANCE_SIZE_MEAN"]<0, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], data["BURO_MONTHS_BALANCE_SIZE_MEAN"] )) + (-1.0))>0, -2.0, (((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (np.tanh((np.tanh((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]))))))*1.) )) * 2.0)) 
    v["i205"] = 0.096020*np.tanh((((((((data["ACTIVE_AMT_ANNUITY_MEAN"]) > ((((data["ACTIVE_AMT_ANNUITY_MAX"]) > (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))*1.)))*1.)) * 2.0)) - (np.maximum((((((data["ACTIVE_AMT_ANNUITY_MEAN"]) > (data["ACTIVE_AMT_ANNUITY_MAX"]))*1.))), (((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (data["ACTIVE_AMT_ANNUITY_MAX"]))*1.))))))) 
    v["i206"] = 0.023997*np.tanh(((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.318310, (((np.tanh((np.tanh((np.tanh((data["ACTIVE_DAYS_CREDIT_VAR"]))))))) > (0.318310))*1.) )) / 2.0)) - ((((3.141593) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.)))) 
    v["i207"] = 0.098400*np.tanh((-1.0*((((((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (((data["NAME_INCOME_TYPE_Student"]) + (data["NAME_EDUCATION_TYPE_Academic_degree"]))))) / 2.0)) + (((((data["NAME_TYPE_SUITE_Group_of_people"]) * 2.0)) + (((data["ORGANIZATION_TYPE_Trade__type_6"]) - (data["ORGANIZATION_TYPE_Industry__type_4"])))))))))) 
    v["i208"] = 0.095998*np.tanh(((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.where((((((np.maximum(((data["AMT_ANNUITY"])), ((3.141593)))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))*1.)>0, data["NAME_INCOME_TYPE_Student"], (5.0) )))) 
    v["i209"] = 0.000001*np.tanh(np.where(((-2.0) + (data["ACTIVE_DAYS_CREDIT_VAR"]))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.where(data["ACTIVE_DAYS_CREDIT_VAR"]>0, (-1.0*(((((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))), (((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.) ) )) 
    v["i210"] = 0.099010*np.tanh(((((data["AMT_ANNUITY"]) - (((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.636620, np.minimum(((np.minimum(((data["ACTIVE_DAYS_CREDIT_VAR"])), ((0.0))))), ((data["NAME_EDUCATION_TYPE_Academic_degree"]))) )) * 2.0)) * 2.0)))) * (data["NAME_TYPE_SUITE_Other_A"]))) 
    v["i211"] = 0.099733*np.tanh(((((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) + (data["CLOSED_MONTHS_BALANCE_SIZE_SUM"]))/2.0)) > (0.318310))*1.)) * (((((((1.0) - (((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) / 2.0)))) - (((data["CLOSED_MONTHS_BALANCE_SIZE_SUM"]) / 2.0)))) * 2.0)))) 
    v["i212"] = 0.096050*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"]>0, ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0), (-1.0*(((((-1.0) > (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))*1.)))) ), (-1.0*(((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) > (1.0))*1.)))) )) 
    v["i213"] = 0.099993*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_MEAN"] < -99998, ((data["EXT_SOURCE_3"]) - (data["AMT_REQ_CREDIT_BUREAU_WEEK"])), ((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) - (3.141593))) * (data["BURO_AMT_CREDIT_SUM_MEAN"]))) * ((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (data["BURO_AMT_CREDIT_SUM_MEAN"]))*1.))) )) 
    v["i214"] = 0.099000*np.tanh(np.where((((-2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)>0, 1.570796, (-1.0*(((((data["EXT_SOURCE_3"]) > (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]>0, (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (1.570796))*1.), 1.570796 )))*1.)))) )) 
    v["i215"] = 0.099496*np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], np.where((((2.0) < (((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["EXT_SOURCE_3"]))))*1.)>0, (((4.0)) * 2.0), ((data["AMT_REQ_CREDIT_BUREAU_DAY"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"])) ) )) 
    v["i216"] = 0.052589*np.tanh(((((((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) - ((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"]))/2.0)))) * ((-1.0*(((((data["EXT_SOURCE_3"]) > ((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (0.0))*1.)))*1.))))))) * 2.0)) * 2.0)) 
    v["i217"] = 0.099605*np.tanh(((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((((np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], ((((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) + (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]))) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) / 2.0))) )) > (data["ACTIVE_AMT_CREDIT_SUM_MEAN"]))*1.))))) * 2.0)) 
    v["i218"] = 0.099979*np.tanh(np.where(data["REFUSED_DAYS_DECISION_MAX"]>0, (-1.0*(((((data["REFUSED_CNT_PAYMENT_SUM"]) > (data["REFUSED_AMT_CREDIT_MEAN"]))*1.)))), (-1.0*(((((data["REFUSED_CNT_PAYMENT_SUM"]) > (0.318310))*1.)))) )) 
    v["i219"] = 0.089880*np.tanh(((((((np.minimum(((data["REFUSED_CNT_PAYMENT_SUM"])), ((np.where(((data["REFUSED_DAYS_DECISION_MEAN"]) * 2.0)<0, data["REFUSED_HOUR_APPR_PROCESS_START_MAX"], -1.0 ))))) > (data["REFUSED_HOUR_APPR_PROCESS_START_MEAN"]))*1.)) > ((((data["REFUSED_DAYS_DECISION_MEAN"]) < ((-1.0*((data["REFUSED_DAYS_DECISION_MAX"])))))*1.)))*1.)) 
    v["i220"] = 0.096999*np.tanh((-1.0*((np.where(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - ((((data["NEW_SCORES_STD"]) < (data["ACTIVE_AMT_ANNUITY_MAX"]))*1.)))>0, data["NEW_SCORES_STD"], ((((-1.0*((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) < ((((data["ORGANIZATION_TYPE_Advertising"]) < (data["ACTIVE_AMT_ANNUITY_MAX"]))*1.)))*1.) ))))) 
    v["i221"] = 0.098925*np.tanh(((((((((((np.maximum(((data["BURO_MONTHS_BALANCE_MIN_MIN"])), ((data["BURO_STATUS_1_MEAN_MEAN"])))) > (data["BURO_STATUS_C_MEAN_MEAN"]))*1.)) < ((((data["BURO_STATUS_1_MEAN_MEAN"]) + (((0.318310) * (((data["BURO_MONTHS_BALANCE_MIN_MIN"]) * 2.0)))))/2.0)))*1.)) * 2.0)) * 2.0)) 
    v["i222"] = 0.099960*np.tanh(((np.minimum(((data["ORGANIZATION_TYPE_Advertising"])), ((((((data["NAME_FAMILY_STATUS_Civil_marriage"]) + (np.where(data["AMT_ANNUITY"]<0, data["AMT_ANNUITY"], ((data["NAME_FAMILY_STATUS_Civil_marriage"]) + (((data["AMT_ANNUITY"]) * 2.0))) )))) * (data["ORGANIZATION_TYPE_Advertising"])))))) * 2.0)) 
    v["i223"] = 0.096983*np.tanh(np.minimum(((((np.tanh(((((data["HOUR_APPR_PROCESS_START"]) > (((np.tanh((data["AMT_ANNUITY"]))) / 2.0)))*1.)))) / 2.0))), (((((((((((3.0)) / 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) > (data["HOUR_APPR_PROCESS_START"]))*1.))))) 
    v["i224"] = 0.070000*np.tanh(((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (data["DAYS_BIRTH"]))) > (0.318310))*1.)) + (np.minimum(((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((data["AMT_ANNUITY"]))))), (((((data["DAYS_BIRTH"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.))))))/2.0)) / 2.0)) 
    v["i225"] = 0.091985*np.tanh(((np.where(data["OCCUPATION_TYPE_Secretaries"]>0, data["AMT_ANNUITY"], (((3.141593) < (data["AMT_ANNUITY"]))*1.) )) + (np.where(data["AMT_ANNUITY"]<0, data["OCCUPATION_TYPE_Secretaries"], np.where(data["OCCUPATION_TYPE_Secretaries"]>0, -1.0, data["ORGANIZATION_TYPE_Advertising"] ) )))) 
    v["i226"] = 0.095139*np.tanh((-1.0*(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["NAME_EDUCATION_TYPE_Academic_degree"], (-1.0*((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)))) )) + ((((0.318310) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) * 2.0)))*1.))))) 
    v["i227"] = 0.098604*np.tanh((((np.minimum(((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["ORGANIZATION_TYPE_Industry__type_11"], (4.0) )) * (((data["ORGANIZATION_TYPE_Industry__type_11"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) > (data["ORGANIZATION_TYPE_Industry__type_11"]))*1.)) 
    v["i228"] = 0.100000*np.tanh((((((((data["AMT_ANNUITY"]) + (((np.where(data["DAYS_BIRTH"]<0, 0.0, (((2.0)) + (data["AMT_ANNUITY"])) )) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) > ((6.81929636001586914)))*1.)) * (data["AMT_ANNUITY"]))) 
    v["i229"] = 0.099554*np.tanh((((((np.where(data["AMT_ANNUITY"]<0, data["NAME_HOUSING_TYPE_With_parents"], ((((-1.0*((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (6.0), 1.570796 ))))) + (1.570796))/2.0) )) - (-2.0))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) 
    v["i230"] = 0.098967*np.tanh(((data["DAYS_BIRTH"]) * ((-1.0*((((0.318310) * (((data["NAME_HOUSING_TYPE_With_parents"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.minimum((((-1.0*((data["AMT_ANNUITY"]))))), ((0.0)))))*1.))))))))))) 
    v["i231"] = 0.061296*np.tanh(((((np.maximum(((data["ORGANIZATION_TYPE_Industry__type_11"])), (((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["AMT_ANNUITY"]))/2.0))))) * (np.minimum((((((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["AMT_ANNUITY"]))/2.0)) / 2.0))), ((data["NAME_HOUSING_TYPE_With_parents"])))))) * (data["AMT_ANNUITY"]))) 
    v["i232"] = 0.098300*np.tanh(np.where(data["NAME_HOUSING_TYPE_With_parents"]<0, ((data["NAME_HOUSING_TYPE_With_parents"]) / 2.0), ((((((data["AMT_ANNUITY"]) + ((((data["AMT_ANNUITY"]) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))*1.)))/2.0)) > ((((0.318310) < (data["AMT_ANNUITY"]))*1.)))*1.) )) 
    v["i233"] = 0.098259*np.tanh(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (((data["WEEKDAY_APPR_PROCESS_START_FRIDAY"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["ORGANIZATION_TYPE_Transport__type_3"]))))/2.0), data["ORGANIZATION_TYPE_Transport__type_3"] )))) - (data["ORGANIZATION_TYPE_Trade__type_5"]))) 
    v["i234"] = 0.099996*np.tanh((((((-1.0*((np.where((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (1.570796))*1.)>0, ((((data["AMT_ANNUITY"]) * 2.0)) * 2.0), np.where(data["AMT_ANNUITY"]<0, data["ORGANIZATION_TYPE_Transport__type_1"], data["ORGANIZATION_TYPE_Trade__type_4"] ) ))))) * 2.0)) * 2.0)) 
    v["i235"] = 0.099031*np.tanh(np.where(np.where(data["OCCUPATION_TYPE_Security_staff"]<0, data["NEW_CAR_TO_BIRTH_RATIO"], data["NAME_EDUCATION_TYPE_Academic_degree"] ) < -99998, data["OCCUPATION_TYPE_Security_staff"], (((data["AMT_ANNUITY"]) + (((data["NEW_CAR_TO_BIRTH_RATIO"]) + (np.where(data["AMT_ANNUITY"]<0, data["NEW_CAR_TO_BIRTH_RATIO"], data["OCCUPATION_TYPE_Security_staff"] )))))/2.0) )) 
    v["i236"] = 0.098701*np.tanh(((((((((np.where(data["AMT_ANNUITY"] < -99998, data["AMT_ANNUITY"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((((((0.636620) * 2.0)) * 2.0)) < (data["AMT_ANNUITY"]))*1.))) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i237"] = 0.085625*np.tanh(((((((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((-1.0*((data["ORGANIZATION_TYPE_Trade__type_4"])))) * 2.0))))) / 2.0)) + ((((((np.tanh((data["OCCUPATION_TYPE_Waiters_barmen_staff"]))) + (-1.0))) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) * 2.0)) 
    v["i238"] = 0.098000*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (((-2.0) > (data["NEW_DOC_IND_KURT"]))*1.), np.where(data["NEW_DOC_IND_KURT"]>0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["REGION_POPULATION_RELATIVE"]))*1.), ((((-2.0) * (data["REGION_POPULATION_RELATIVE"]))) / 2.0) ) )) 
    v["i239"] = 0.094999*np.tanh(np.where(data["WALLSMATERIAL_MODE_Wooden"]<0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, np.where(data["WEEKDAY_APPR_PROCESS_START_MONDAY"]<0, data["NAME_INCOME_TYPE_Student"], data["REGION_POPULATION_RELATIVE"] ), (((data["REGION_POPULATION_RELATIVE"]) > (3.0))*1.) ), np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["REGION_POPULATION_RELATIVE"]))) )) 
    v["i240"] = 0.078810*np.tanh(np.where(data["NEW_DOC_IND_KURT"]<0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((data["OCCUPATION_TYPE_Medicine_staff"]) - (data["AMT_ANNUITY"])), data["AMT_CREDIT"] ), (-1.0*(((((data["AMT_CREDIT"]) > ((2.0)))*1.)))) )) 
    v["i241"] = 0.099910*np.tanh(((((((((((np.tanh((1.570796))) < (data["CC_COUNT"]))*1.)) > (((((data["CC_COUNT"]) - (data["CC_SK_DPD_DEF_SUM"]))) * ((((data["NAME_INCOME_TYPE_Student"]) < (data["CC_SK_DPD_DEF_SUM"]))*1.)))))*1.)) * 2.0)) * 2.0)) 
    v["i242"] = 0.094993*np.tanh(((data["CC_AMT_PAYMENT_CURRENT_VAR"]) - (np.where((((np.minimum(((data["CC_AMT_PAYMENT_CURRENT_MEAN"])), ((data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"])))) > (data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]))*1.)>0, 2.0, ((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) * 2.0)) - (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"])) )))) 
    v["i243"] = 0.098760*np.tanh(np.minimum(((np.maximum(((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"] < -99998, ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"])), data["CC_AMT_PAYMENT_CURRENT_MEAN"] ))), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))))), ((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) * (((data["CC_AMT_PAYMENT_CURRENT_VAR"]) - (2.0)))))))) 
    v["i244"] = 0.068305*np.tanh(np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["NAME_CONTRACT_TYPE_Cash_loans"], ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) - ((4.46922969818115234)))) * ((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))) > (((1.570796) - (data["NAME_CONTRACT_TYPE_Cash_loans"]))))*1.))) )) 
    v["i245"] = 0.098999*np.tanh(((data["EXT_SOURCE_3"]) * ((-1.0*((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.))), (((data["EXT_SOURCE_3"]) > (1.570796))*1.) ))))))) 
    v["i246"] = 0.076350*np.tanh(((((data["EXT_SOURCE_3"]) * 2.0)) * ((((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, ((((-1.0*((data["EXT_SOURCE_3"])))) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.), (-1.0*((data["AMT_ANNUITY"]))) )) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.)))) 
    v["i247"] = 0.099104*np.tanh((((((np.tanh((data["EXT_SOURCE_3"]))) > ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0)))*1.)) - ((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) < ((-1.0*((data["EXT_SOURCE_3"])))))*1.)) + (data["NEW_EXT_SOURCES_MEAN"]))))) 
    v["i248"] = 0.090002*np.tanh(((((data["DAYS_BIRTH"]) + ((((-1.0) + (data["BURO_CREDIT_TYPE_Car_loan_MEAN"]))/2.0)))) * (np.minimum(((((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) - (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))), (((((((data["NEW_SCORES_STD"]) * 2.0)) > (data["DAYS_BIRTH"]))*1.))))))) 
    v["i249"] = 0.099966*np.tanh((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) - ((((((((np.tanh((((np.tanh((((((data["DAYS_BIRTH"]) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * (data["CODE_GENDER"]))) + (-1.0))/2.0))))))) 
    v["i250"] = 0.098702*np.tanh(((((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (data["FLAG_EMP_PHONE"]))*1.)) > ((((0.636620) + (((0.636620) + (np.minimum(((data["REGION_RATING_CLIENT"])), ((data["DAYS_BIRTH"])))))))/2.0)))*1.)) 
    v["i251"] = 0.099990*np.tanh((-1.0*((np.where((-1.0*((data["NEW_SCORES_STD"])))>0, data["REGION_RATING_CLIENT"], ((data["NEW_EXT_SOURCES_MEAN"]) + ((((data["NEW_SCORES_STD"]) > ((((((data["NEW_EXT_SOURCES_MEAN"]) + (0.636620))) > (data["REGION_RATING_CLIENT"]))*1.)))*1.))) ))))) 
    v["i252"] = 0.099230*np.tanh(((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]>0, np.minimum((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) < (((data["REG_CITY_NOT_WORK_CITY"]) / 2.0)))*1.))), ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["REGION_RATING_CLIENT"]))))), (-1.0*((((data["REG_CITY_NOT_WORK_CITY"]) / 2.0)))) )) / 2.0)) 
    v["i253"] = 0.099500*np.tanh(((((data["NEW_EXT_SOURCES_MEAN"]) + ((13.18767356872558594)))) * ((((np.where(data["NEW_EXT_SOURCES_MEAN"]>0, ((((((data["NEW_EXT_SOURCES_MEAN"]) + (-1.0))/2.0)) < (data["CLOSED_MONTHS_BALANCE_SIZE_SUM"]))*1.), -2.0 )) > (data["NEW_EXT_SOURCES_MEAN"]))*1.)))) 
    v["i254"] = 0.099742*np.tanh(((((3.0) - (np.where(np.where(data["EXT_SOURCE_1"] < -99998, ((3.0) - (data["ACTIVE_AMT_ANNUITY_MEAN"])), data["DAYS_BIRTH"] )<0, data["EXT_SOURCE_1"], 3.141593 )))) * 2.0)) 
    v["i255"] = 0.099548*np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_SOURCES_PROD"], ((np.tanh((np.tanh((data["NEW_EXT_SOURCES_MEAN"]))))) - (data["NEW_EXT_SOURCES_MEAN"])) ), np.minimum(((0.636620)), ((data["NEW_EXT_SOURCES_MEAN"]))) )) 
    v["i256"] = 0.089995*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, np.minimum(((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])))), ((data["AMT_REQ_CREDIT_BUREAU_QRT"]))), (((data["EXT_SOURCE_3"]) > (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) - (np.minimum(((data["AMT_REQ_CREDIT_BUREAU_YEAR"])), ((-1.0)))))))*1.) )) 
    v["i257"] = 0.099999*np.tanh(((np.tanh((np.minimum(((((np.where(data["EXT_SOURCE_3"] < -99998, data["HOUR_APPR_PROCESS_START"], np.maximum(((data["NEW_INC_BY_ORG"])), ((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))) )) + (data["HOUR_APPR_PROCESS_START"])))), ((((data["EXT_SOURCE_3"]) * (data["NEW_INC_BY_ORG"])))))))) / 2.0)) 
    v["i258"] = 0.099998*np.tanh(np.minimum(((0.318310)), ((((((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) > ((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) < ((((data["ACTIVE_AMT_ANNUITY_MAX"]) < ((-1.0*((data["CLOSED_MONTHS_BALANCE_MAX_MAX"])))))*1.)))*1.)))*1.)) < ((((data["NEW_INC_BY_ORG"]) > (data["OCCUPATION_TYPE_Cooking_staff"]))*1.)))*1.))))) 
    v["i259"] = 0.044980*np.tanh(np.maximum(((np.where(data["LIVINGAREA_MODE"]<0, data["OCCUPATION_TYPE_Medicine_staff"], (((np.where(data["DAYS_ID_PUBLISH"]<0, 1.570796, data["OCCUPATION_TYPE_High_skill_tech_staff"] )) > (data["OCCUPATION_TYPE_Medicine_staff"]))*1.) ))), ((((-2.0) - (data["NEW_EXT_SOURCES_MEAN"])))))) 
    v["i260"] = 0.099700*np.tanh(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, ((data["AMT_INCOME_TOTAL"]) * 2.0), np.where((((data["DAYS_ID_PUBLISH"]) + (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))/2.0)<0, ((0.636620) / 2.0), np.minimum(((data["OCCUPATION_TYPE_High_skill_tech_staff"])), (((-1.0*((data["AMT_INCOME_TOTAL"])))))) ) )) 
    v["i261"] = 0.099990*np.tanh((((((((data["NEW_EXT_SOURCES_MEAN"]) < (((np.tanh((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))) - (((((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) - (data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))))*1.)) + (3.141593))/2.0)))))*1.)) * 2.0)) * 2.0)) 
    v["i262"] = 0.094001*np.tanh(np.minimum((((((data["NEW_EXT_SOURCES_MEAN"]) > (0.636620))*1.))), (((((11.55127620697021484)) * ((((((((0.636620) > (((data["EXT_SOURCE_1"]) / 2.0)))*1.)) / 2.0)) - (((data["NEW_EXT_SOURCES_MEAN"]) / 2.0))))))))) 
    v["i263"] = 0.099700*np.tanh((-1.0*(((((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) > (np.where(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["BURO_CREDIT_TYPE_Loan_for_business_development_MEAN"], 3.141593 )>0, (-1.0*((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))), ((3.141593) + (0.636620)) )))*1.))))) 
    v["i264"] = 0.096890*np.tanh(((np.minimum(((np.minimum(((((data["NAME_EDUCATION_TYPE_Higher_education"]) / 2.0))), ((data["NEW_EXT_SOURCES_MEAN"]))))), ((data["AMT_INCOME_TOTAL"])))) * ((-1.0*(((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]<0, data["AMT_INCOME_TOTAL"], 3.141593 )) + (data["NAME_EDUCATION_TYPE_Higher_education"]))/2.0))))))) 
    v["i265"] = 0.099603*np.tanh(((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, np.tanh((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) * 2.0))), ((data["CODE_GENDER"]) + (((((data["NEW_CAR_TO_EMPLOY_RATIO"]) + (data["NEW_CAR_TO_EMPLOY_RATIO"]))) + (data["NEW_CAR_TO_EMPLOY_RATIO"])))) )) / 2.0)) / 2.0)) 
    v["i266"] = 0.083000*np.tanh(np.maximum((((((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (data["NEW_EXT_SOURCES_MEAN"]))*1.)) * (data["NEW_EXT_SOURCES_MEAN"])))), ((np.where(data["NEW_EXT_SOURCES_MEAN"]>0, np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, data["EXT_SOURCE_1"], data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] ), ((data["EXT_SOURCE_1"]) * 2.0) ))))) 
    v["i267"] = 0.097003*np.tanh(np.minimum(((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * ((((data["OCCUPATION_TYPE_Cooking_staff"]) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))/2.0))))), (((((((((-1.0*((data["NAME_TYPE_SUITE_Other_A"])))) * (data["OCCUPATION_TYPE_Cooking_staff"]))) - (data["NAME_TYPE_SUITE_Other_A"]))) - (data["ORGANIZATION_TYPE_Trade__type_4"])))))) 
    v["i268"] = 0.097000*np.tanh(((((((np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["WEEKDAY_APPR_PROCESS_START_TUESDAY"], (((0.636620) > (((3.0) + (((data["NEW_EXT_SOURCES_MEAN"]) + (data["NAME_INCOME_TYPE_Maternity_leave"]))))))*1.) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i269"] = 0.076496*np.tanh(((((data["FLAG_DOCUMENT_3"]) * (np.minimum(((0.0)), ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (((((-1.0) - (data["AMT_ANNUITY"]))) + (np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))/2.0), 0.636620 ))))))) * 2.0)) 
    v["i270"] = 0.099960*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (((data["NEW_EXT_SOURCES_MEAN"]) < (-1.0))*1.), ((np.where(data["NEW_EXT_SOURCES_MEAN"]<0, data["ORGANIZATION_TYPE_Trade__type_3"], (-1.0*((data["ORGANIZATION_TYPE_Trade__type_3"]))) )) - ((((1.570796) < (data["NEW_EXT_SOURCES_MEAN"]))*1.))) )) 
    v["i271"] = 0.096900*np.tanh(np.where(data["BURO_AMT_ANNUITY_MAX"]<0, np.where(data["ORGANIZATION_TYPE_Trade__type_3"]<0, (((0.636620) < (((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]) / 2.0)))*1.), ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) * 2.0)) * 2.0) ), (((data["BURO_AMT_ANNUITY_MAX"]) > (data["NEW_EXT_SOURCES_MEAN"]))*1.) )) 
    v["i272"] = 0.056000*np.tanh((((((((((-1.0*((np.maximum(((np.where(data["EXT_SOURCE_1"]<0, (-1.0*((np.maximum(((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])), ((data["NAME_EDUCATION_TYPE_Academic_degree"])))))), data["APPROVED_AMT_DOWN_PAYMENT_MEAN"] ))), ((data["NAME_EDUCATION_TYPE_Academic_degree"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i273"] = 0.095900*np.tanh(np.maximum((((((((1.570796) < (((data["NEW_CAR_TO_BIRTH_RATIO"]) * 2.0)))*1.)) * ((-1.0*((data["NEW_EXT_SOURCES_MEAN"]))))))), ((np.where(((data["NEW_EXT_SOURCES_MEAN"]) - (1.570796))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], -2.0 ))))) 
    v["i274"] = 0.065300*np.tanh((((((((((((data["NEW_CAR_TO_BIRTH_RATIO"]) < ((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["OWN_CAR_AGE"]))*1.)))*1.)) > (data["NEW_EXT_SOURCES_MEAN"]))*1.)) < (((data["OWN_CAR_AGE"]) * (((1.570796) / 2.0)))))*1.)) * (1.570796))) 
    v["i275"] = 0.098970*np.tanh(((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((0.636620) < (((data["NEW_CAR_TO_BIRTH_RATIO"]) - (((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((data["NEW_CAR_TO_BIRTH_RATIO"]) < (((data["NEW_EXT_SOURCES_MEAN"]) * (data["NEW_EXT_SOURCES_MEAN"]))))*1.)))))))*1.)))) 
    v["i276"] = 0.050995*np.tanh((((((((((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) > (((data["AMT_ANNUITY"]) * (data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]))))*1.)) - (data["NAME_INCOME_TYPE_Student"]))) - (np.where(data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]<0, data["NAME_INCOME_TYPE_Student"], data["AMT_ANNUITY"] )))) / 2.0)) / 2.0)) 
    v["i277"] = 0.099600*np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], ((((((-1.0) < (np.where(data["OWN_CAR_AGE"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CAR_TO_BIRTH_RATIO"]))) )))*1.)) < (((data["NEW_EXT_SOURCES_MEAN"]) / 2.0)))*1.) )) 
    v["i278"] = 0.098097*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((3.0) * (np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (3.0))*1.)>0, data["NEW_EXT_SOURCES_MEAN"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["NEW_EXT_SOURCES_MEAN"]) + (3.141593))))*1.) )))))) 
    v["i279"] = 0.099890*np.tanh(np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], np.minimum(((((((0.318310) * 2.0)) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])))), (((((-1.0*(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["NEW_CAR_TO_EMPLOY_RATIO"]))*1.))))) / 2.0)))) )) 
    v["i280"] = 0.093509*np.tanh(np.where(((np.tanh((np.tanh((data["OWN_CAR_AGE"]))))) - (data["CNT_CHILDREN"]))<0, (((data["FLAG_OWN_CAR"]) > (data["CNT_CHILDREN"]))*1.), np.tanh((np.tanh((np.tanh((np.tanh((data["OWN_CAR_AGE"])))))))) )) 
    v["i281"] = 0.091999*np.tanh((((((((data["AMT_ANNUITY"]) > ((2.73518514633178711)))*1.)) * 2.0)) + (((np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], (((data["NEW_EXT_SOURCES_MEAN"]) < ((-1.0*(((2.73518514633178711))))))*1.) )) + (data["ORGANIZATION_TYPE_Telecom"]))))) 
    v["i282"] = 0.080945*np.tanh(np.where(np.where(((((data["AMT_ANNUITY"]) * (np.minimum(((data["AMT_ANNUITY"])), (((-1.0*((data["NAME_EDUCATION_TYPE_Academic_degree"]))))))))) + (data["NAME_INCOME_TYPE_Maternity_leave"]))>0, data["ORGANIZATION_TYPE_Trade__type_4"], 3.0 )<0, data["ORGANIZATION_TYPE_Trade__type_4"], data["WEEKDAY_APPR_PROCESS_START_TUESDAY"] )) 
    v["i283"] = 0.098110*np.tanh(((((((data["AMT_ANNUITY"]) + ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) + (np.minimum(((data["NAME_EDUCATION_TYPE_Academic_degree"])), ((data["EMERGENCYSTATE_MODE_Yes"])))))) * (((data["ORGANIZATION_TYPE_Transport__type_2"]) - (data["EMERGENCYSTATE_MODE_Yes"]))))) 
    v["i284"] = 0.059900*np.tanh((((np.maximum(((((((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((((np.tanh((((data["ORGANIZATION_TYPE_Advertising"]) / 2.0)))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (1.570796))*1.)) 
    v["i285"] = 0.088980*np.tanh(np.minimum(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["INSTAL_PAYMENT_PERC_MEAN"])))), (((-1.0*((np.where(data["INSTAL_PAYMENT_PERC_MEAN"]>0, 3.141593, (((data["INSTAL_PAYMENT_PERC_MEAN"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["INSTAL_PAYMENT_PERC_MEAN"]))))/2.0) )))))))) 
    v["i286"] = 0.058000*np.tanh(np.minimum(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((data["ORGANIZATION_TYPE_Legal_Services"]) / 2.0), data["OCCUPATION_TYPE_High_skill_tech_staff"] ))), (((-1.0*((((((((-2.0) * 2.0)) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))) * 2.0)))))))) 
    v["i287"] = 0.099990*np.tanh((((((((-1.0*((((((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)) < (((0.318310) + (data["NAME_INCOME_TYPE_Maternity_leave"]))))*1.))))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) 
    v["i288"] = 0.089938*np.tanh(((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((np.minimum(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NAME_INCOME_TYPE_Maternity_leave"]))/2.0)) / 2.0), data["AMT_ANNUITY"] ))), ((data["NAME_INCOME_TYPE_Maternity_leave"]))))))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) 
    v["i289"] = 0.099996*np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, ((data["AMT_ANNUITY"]) - ((((data["AMT_ANNUITY"]) < (((2.0) / 2.0)))*1.))), (((data["AMT_ANNUITY"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.) )) * (((data["DAYS_REGISTRATION"]) / 2.0)))) 
    v["i290"] = 0.099996*np.tanh(((((((((data["WALLSMATERIAL_MODE_Stone__brick"]) + ((-1.0*((np.minimum((((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))), ((np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))))))/2.0)) * 2.0)) > (3.0))*1.)) 
    v["i291"] = 0.063400*np.tanh((((-1.0*(((((((np.minimum(((data["NONLIVINGAREA_MEDI"])), (((-1.0*((data["NONLIVINGAREA_AVG"]))))))) > (((data["NAME_INCOME_TYPE_Maternity_leave"]) - (0.318310))))*1.)) * (((data["WALLSMATERIAL_MODE_Stone__brick"]) / 2.0))))))) * 2.0)) 
    v["i292"] = 0.076349*np.tanh(((data["AMT_ANNUITY"]) * ((-1.0*(((((data["YEARS_BUILD_AVG"]) > ((((data["YEARS_BUILD_AVG"]) + (((((((data["AMT_ANNUITY"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) > (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.)))/2.0)))*1.))))))) 
    v["i293"] = 0.099173*np.tanh((((((data["LIVINGAPARTMENTS_AVG"]) > (np.where((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))>0, (5.0), np.maximum((((-1.0*(((((3.0) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))*1.)))))), ((data["AMT_ANNUITY"]))) )))*1.)) * 2.0)) 
    v["i294"] = 0.099970*np.tanh((-1.0*(((((np.where(data["LIVINGAPARTMENTS_AVG"] < -99998, data["YEARS_BUILD_MODE"], data["COMMONAREA_MEDI"] )) > (np.minimum((((((data["YEARS_BUILD_MODE"]) < ((1.0)))*1.))), ((np.tanh((((data["YEARS_BUILD_MODE"]) * 2.0))))))))*1.))))) 
    v["i295"] = 0.099718*np.tanh((((((data["FLOORSMIN_AVG"]) > (np.where(data["LIVINGAREA_AVG"]>0, np.maximum(((0.636620)), ((data["LIVINGAREA_MODE"]))), ((data["LIVINGAREA_AVG"]) * (data["FLOORSMIN_AVG"])) )))*1.)) * ((((10.97417545318603516)) * (data["LIVINGAREA_AVG"]))))) 
    v["i296"] = 0.079954*np.tanh(np.where(data["WALLSMATERIAL_MODE_Stone__brick"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((np.where((((((data["LIVINGAREA_MEDI"]) * 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)>0, (3.0), data["NEW_ANNUITY_TO_INCOME_RATIO"] )) * 2.0))) )) 
    v["i297"] = 0.098995*np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.minimum(((np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"]))>0, data["ORGANIZATION_TYPE_Advertising"], data["AMT_ANNUITY"] ))), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"]))) - (data["EXT_SOURCE_1"])))))))) 
    v["i298"] = 0.079895*np.tanh(np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]<0, ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0))*1.)))) - (data["WALLSMATERIAL_MODE_Monolithic"]))) - (data["NAME_INCOME_TYPE_Student"])), data["WALLSMATERIAL_MODE_Monolithic"] )) 
    v["i299"] = 0.099500*np.tanh((((((-2.0) > (np.where(data["EXT_SOURCE_3"] < -99998, ((-2.0) * (data["AMT_ANNUITY"])), ((np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), (((((-1.0*((data["EXT_SOURCE_3"])))) * 2.0))))) * 2.0) )))*1.)) * 2.0)) 
    v["i300"] = 0.099751*np.tanh(np.where(data["CLOSED_DAYS_CREDIT_VAR"]>0, np.minimum(((((((((((data["AMT_ANNUITY"]) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) * 2.0)) * 2.0)) * 2.0))), ((((data["BURO_DAYS_CREDIT_VAR"]) * (data["AMT_ANNUITY"]))))), (0.0) )) 
    v["i301"] = 0.057002*np.tanh(((((((((data["BURO_DAYS_CREDIT_VAR"]) / 2.0)) < (data["EXT_SOURCE_3"]))*1.)) < (((((-1.0*((0.318310)))) < (np.minimum(((((data["BURO_DAYS_CREDIT_VAR"]) * 2.0))), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])))))*1.)))*1.)) 
    v["i302"] = 0.094999*np.tanh((((((((-1.0*((np.maximum(((data["CLOSED_DAYS_CREDIT_VAR"])), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))))))) < (((1.0) - (np.minimum(((0.636620)), ((1.0)))))))*1.)) < ((((data["EXT_SOURCE_3"]) > (data["BURO_DAYS_CREDIT_VAR"]))*1.)))*1.)) 
    v["i303"] = 0.097200*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_MEAN"]<0, np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]<0, ((((1.09531068801879883)) < (((data["ACTIVE_DAYS_CREDIT_MEAN"]) + ((((data["ACTIVE_AMT_CREDIT_SUM_MEAN"]) < (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)))))*1.), data["ACTIVE_DAYS_CREDIT_MEAN"] ), ((data["ACTIVE_DAYS_CREDIT_MEAN"]) * 2.0) )) 
    v["i304"] = 0.099712*np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"]>0, np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, -2.0, ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) / 2.0)) * (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) ), np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"], (-1.0*((0.318310))) ) )) 
    v["i305"] = 0.099669*np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))*1.)))) * ((((data["CLOSED_DAYS_CREDIT_VAR"]) > (np.minimum(((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * 2.0))), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))*1.)))) 
    v["i306"] = 0.024020*np.tanh(np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.))), ((((np.minimum(((-2.0)), ((((-2.0) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))))) * (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))))) 
    v["i307"] = 0.097699*np.tanh((((((((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.)) / 2.0)) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))) * 2.0)) * (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((((((-1.0*((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))) < (data["CLOSED_DAYS_CREDIT_VAR"]))*1.))))))) 
    v["i308"] = 0.095950*np.tanh(((np.where(data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]>0, np.where(data["BURO_DAYS_CREDIT_MEAN"]<0, 0.0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] ), (-1.0*((((((data["BURO_DAYS_CREDIT_MEAN"]) - (data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]))) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))) )) / 2.0)) 
    v["i309"] = 0.068960*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"], ((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, (0.06772400438785553), (-1.0*((((((-1.0*(((((0.06772400438785553)) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))))) < (data["ACTIVE_AMT_CREDIT_SUM_MEAN"]))*1.)))) )) * 2.0) )) 
    v["i310"] = 0.097900*np.tanh((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (((np.tanh((np.tanh((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))))) / 2.0)))*1.)) * (((((((3.141593) - (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))) - (data["CLOSED_AMT_CREDIT_SUM_SUM"]))) - (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))))) 
    v["i311"] = 0.058059*np.tanh(np.where((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))/2.0)>0, (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (data["AMT_ANNUITY"]))/2.0), np.where(((0.318310) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0)))>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["NAME_INCOME_TYPE_Maternity_leave"] ) )) 
    v["i312"] = 0.004031*np.tanh((((((((0.636620) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)) * (((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - ((((0.636620) > (data["AMT_ANNUITY"]))*1.)))) - ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (data["AMT_CREDIT"]))*1.)))))) * 2.0)) 
    v["i313"] = 0.029029*np.tanh((((np.where((-1.0*((np.maximum(((data["AMT_ANNUITY"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["AMT_CREDIT"]))) )) > ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((-1.0*((1.0)))))*1.)))*1.)) 
    v["i314"] = 0.051999*np.tanh(((np.minimum(((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]<0, np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, ((1.0) - (data["AMT_CREDIT"])), data["NEW_ANNUITY_TO_INCOME_RATIO"] ), data["NEW_ANNUITY_TO_INCOME_RATIO"] ))), (((((data["AMT_CREDIT"]) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.))))) / 2.0)) 
    v["i315"] = 0.099950*np.tanh(((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]>0, (-1.0*((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))), ((((((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) + (np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]>0, data["NAME_INCOME_TYPE_Maternity_leave"], 2.0 )))/2.0)) < (((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0)))*1.) )) * 2.0)) 
    v["i316"] = 0.100000*np.tanh(((np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]>0, (((-1.0*((((((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]) - (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) - (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])))))) * ((8.35832691192626953))), (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) > ((5.0)))*1.) )) * 2.0)) 
    v["i317"] = 0.090700*np.tanh(np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]>0, data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] )<0, (((data["ACTIVE_DAYS_CREDIT_MEAN"]) > (((((2.37051534652709961)) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))/2.0)))*1.), ((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0))) )) 
    v["i318"] = 0.071000*np.tanh(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]>0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"], np.where((((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))/2.0)<0, 0.0, ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0)) - ((((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))/2.0))) ) )) 
    v["i319"] = 0.096003*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]>0, data["ORGANIZATION_TYPE_Trade__type_4"], ((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (0.318310)) )) 
    v["i320"] = 0.099110*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]<0, np.where(data["ACTIVE_DAYS_CREDIT_MEAN"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] ), ((((((((((np.tanh((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) * 2.0)) + (data["ACTIVE_DAYS_CREDIT_MEAN"]))) * 2.0)) * 2.0)) * 2.0) )) 
    v["i321"] = 0.099160*np.tanh((((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) > (((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))*1.)) - (((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]>0, 0.318310, data["ACTIVE_DAYS_CREDIT_MEAN"] )))))) < (3.0))*1.)))*1.)) 
    v["i322"] = 0.083290*np.tanh((((np.where(data["ACTIVE_DAYS_CREDIT_MEAN"]<0, (((data["ACTIVE_DAYS_CREDIT_MEAN"]) > (np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]<0, data["ACTIVE_DAYS_CREDIT_MEAN"], data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"] )))*1.), data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] )) > (np.tanh((((1.0) / 2.0)))))*1.)) 
    v["i323"] = 0.095882*np.tanh(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]>0, np.maximum(((data["ACTIVE_DAYS_CREDIT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))), ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0)) * (np.maximum(((np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] < -99998, 0.0, data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] ))), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))))) )) 
    v["i324"] = 0.099008*np.tanh(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) * (np.maximum(((((((((-2.0) + ((((-2.0) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))/2.0)))/2.0)) + (data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]))/2.0))), ((((data["ACTIVE_DAYS_CREDIT_MEAN"]) - (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])))))))) 
    v["i325"] = 0.001009*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]<0, np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] < -99998, ((0.318310) / 2.0), data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] ), ((((((((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0) )) 
    v["i326"] = 0.088000*np.tanh(np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] )>0, ((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), (-1.0*((((((2.65623164176940918)) < (((0.0) + (data["AMT_CREDIT"]))))*1.)))) )) 
    v["i327"] = 0.013070*np.tanh(np.tanh((((data["ACTIVE_AMT_CREDIT_SUM_MEAN"]) * (((-2.0) * ((((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]<0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"] )) > (np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]>0, 0.0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"] )))*1.)))))))) 
    v["i328"] = 0.099470*np.tanh((((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"])))) < (((np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]>0, ((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), (((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))/2.0) )) * 2.0)))*1.)) 
    v["i329"] = 0.091989*np.tanh(((np.where(-2.0 < -99998, 0.318310, (((((0.318310) < (np.tanh((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))))*1.)) - ((((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) > (np.tanh((-2.0))))*1.))) )) / 2.0)) 
    v["i330"] = 0.096602*np.tanh(((np.where(((np.tanh(((7.0)))) + (data["ACTIVE_DAYS_CREDIT_MEAN"]))>0, ((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * ((7.0))), (((((2.0) + (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))) < (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))*1.) )) * 2.0)) 
    v["i331"] = 0.100000*np.tanh(((np.maximum((((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) - (0.318310))) > (1.570796))*1.))), ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0))))) - ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) > ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)))*1.)))) 
    v["i332"] = 0.091990*np.tanh(np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]>0, -1.0, (((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) < (np.minimum(((np.minimum(((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))))), (((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + (((-1.0) / 2.0)))/2.0))))))*1.)) * 2.0) )) 
    v["i333"] = 0.099995*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]<0, ((((((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)) > (data["ACTIVE_AMT_CREDIT_SUM_SUM"]))*1.)) * ((((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)) * (data["ACTIVE_DAYS_CREDIT_MEAN"])))), data["ACTIVE_DAYS_CREDIT_MEAN"] )) 
    v["i334"] = 0.094326*np.tanh(((((((((np.maximum(((2.0)), ((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"]))))))) > (np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((3.141593)))))*1.)) + (-1.0))/2.0)) * ((6.0)))) 
    v["i335"] = 0.092296*np.tanh((((np.where((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (1.570796))/2.0)>0, (((3.0) < (((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (np.maximum(((data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"])), ((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])))))))*1.), data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"] )) > (0.636620))*1.)) 
    v["i336"] = 0.016198*np.tanh((-1.0*((((np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]<0, (((data["ACTIVE_DAYS_CREDIT_MEAN"]) > (0.318310))*1.), ((((((data["ACTIVE_DAYS_CREDIT_MEAN"]) > (0.318310))*1.)) + ((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) + (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]))/2.0)))/2.0) )) / 2.0))))) 
    v["i337"] = 0.095499*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]>0, (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) < (((((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]) * 2.0)) * 2.0)))*1.), np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((np.where(data["ACTIVE_DAYS_CREDIT_MEAN"]<0, 0.0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] )))) )) 
    v["i338"] = 0.099099*np.tanh(np.where(data["ACTIVE_AMT_ANNUITY_MAX"]>0, data["ACTIVE_CNT_CREDIT_PROLONG_SUM"], (-1.0*(((((((data["REFUSED_AMT_ANNUITY_MEAN"]) < (data["REFUSED_AMT_ANNUITY_MAX"]))*1.)) - ((((((data["REFUSED_AMT_ANNUITY_MEAN"]) < (data["REFUSED_AMT_ANNUITY_MAX"]))*1.)) * ((-1.0*((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])))))))))) )) 
    v["i339"] = 0.098480*np.tanh((((((((np.maximum(((((data["ORGANIZATION_TYPE_Trade__type_4"]) * (((data["ORGANIZATION_TYPE_Trade__type_4"]) + (3.141593)))))), ((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0))))) < (np.minimum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))))*1.)) * 2.0)) * 2.0)) 
    v["i340"] = 0.100000*np.tanh(((((((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, data["AMT_CREDIT"], ((data["NAME_INCOME_TYPE_Student"]) * 2.0) )) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) * (np.maximum(((0.0)), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) * 2.0)) 
    v["i341"] = 0.090000*np.tanh(np.where(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]>0, data["AMT_CREDIT"], (((data["EXT_SOURCE_3"]) > (data["AMT_CREDIT"]))*1.) )>0, 0.318310, (-1.0*(((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > ((((data["NAME_INCOME_TYPE_Student"]) > (data["AMT_CREDIT"]))*1.)))*1.)))) )) 
    v["i342"] = 0.075000*np.tanh(((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((2.53327202796936035)))*1.), ((((1.53511798381805420)) < (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (0.318310))))*1.) )) * ((((5.44602632522583008)) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))) 
    v["i343"] = 0.100000*np.tanh(((np.minimum((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (2.0))*1.)) / 2.0))), (((((8.44388580322265625)) - (np.maximum(((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))))))) * ((((8.44388580322265625)) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))) 
    v["i344"] = 0.050000*np.tanh((-1.0*(((((0.636620) < (np.where((((-1.0*((0.636620)))) + (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))<0, np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((((data["AMT_CREDIT"]) / 2.0)))), (-1.0*((data["AMT_CREDIT"]))) )))*1.))))) 
    v["i345"] = 0.099050*np.tanh((((((-1.0*(((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (-1.0))) > (((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (-1.0))) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)) + ((5.21866846084594727)))/2.0)))*1.))))) * 2.0)) * 2.0)) 
    v["i346"] = 0.095400*np.tanh((-1.0*((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["NAME_INCOME_TYPE_Student"], np.where(data["AMT_ANNUITY"]>0, data["ORGANIZATION_TYPE_Trade__type_4"], np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, 1.570796, data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] ) ) ))))) 
    v["i347"] = 0.092000*np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, (((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"], (-1.0*((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0)))) )) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.), (((3.141593) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.) )) 
    v["i348"] = 0.099980*np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((((5.0)) + (np.minimum(((3.141593)), (((-1.0*((1.570796))))))))/2.0)))*1.)) - ((((3.141593) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (-1.0))))*1.)))) 
    v["i349"] = 0.072006*np.tanh((-1.0*(((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (((data["AMT_ANNUITY"]) + (((((((2.0)) + (0.318310))/2.0)) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) * ((((0.318310) + (data["AMT_ANNUITY"]))/2.0)))))))))*1.))))) 
    v["i350"] = 0.070695*np.tanh(((np.minimum((((((0.636620) < (np.minimum(((data["AMT_ANNUITY"])), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))))*1.))), (((((np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (data["AMT_ANNUITY"])))))) < (1.570796))*1.))))) * 2.0)) 
    v["i351"] = 0.099010*np.tanh(((3.0) * (((3.0) * ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((((3.141593) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))*1.)))) > (3.0))*1.)))))) 
    v["i352"] = 0.001700*np.tanh(np.where(((np.maximum(((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) + (-2.0))<0, 0.0, np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]<0, ((data["AMT_ANNUITY"]) * ((9.0))), data["NEW_CREDIT_TO_ANNUITY_RATIO"] ) )) 
    v["i353"] = 0.037040*np.tanh(np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (-2.0))/2.0)<0, np.where(((data["AMT_ANNUITY"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))>0, data["NAME_INCOME_TYPE_Maternity_leave"], data["ORGANIZATION_TYPE_Realtor"] ), ((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)) * 2.0) )) 
    v["i354"] = 0.089980*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]<0, (((3.0) < ((((((-1.0*((data["ORGANIZATION_TYPE_Trade__type_4"])))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * 2.0)))*1.), np.where(data["NAME_HOUSING_TYPE_Municipal_apartment"]<0, (-1.0*((data["ORGANIZATION_TYPE_Trade__type_4"]))), data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] ) )) 
    v["i355"] = 0.099952*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["NAME_CONTRACT_TYPE_Revolving_loans"]))*1.))), (((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) / 2.0)) < (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])) )) 
    v["i356"] = 0.071280*np.tanh(np.where(data["AMT_ANNUITY"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], np.where(data["NAME_HOUSING_TYPE_Municipal_apartment"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], ((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0)) - (data["AMT_ANNUITY"]))) * ((14.04221153259277344)))) - (data["NAME_HOUSING_TYPE_Municipal_apartment"])) ) )) 
    v["i357"] = 0.067000*np.tanh((((((2.0) < (((((data["AMT_ANNUITY"]) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.)) * (((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))))) 
    v["i358"] = 0.098920*np.tanh((((-1.0*(((((((3.141593) + (((-1.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((((data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["AMT_ANNUITY"]))*1.)))*1.)))*1.))))) * 2.0)) 
    v["i359"] = 0.099730*np.tanh((((((((data["AMT_ANNUITY"]) * 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) > ((((data["REG_CITY_NOT_LIVE_CITY"]) < (((1.570796) + (((data["AMT_ANNUITY"]) * 2.0)))))*1.)))*1.)))) 
    v["i360"] = 0.096000*np.tanh(((data["NAME_INCOME_TYPE_Maternity_leave"]) - (((((((((((((((data["AMT_ANNUITY"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)) > (data["AMT_INCOME_TOTAL"]))*1.)) > (data["AMT_ANNUITY"]))*1.)) + (data["AMT_INCOME_TOTAL"]))) > (1.570796))*1.)))) 
    v["i361"] = 0.098949*np.tanh(((((((data["OCCUPATION_TYPE_Managers"]) * (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), (((((-1.0) < (data["AMT_ANNUITY"]))*1.))))))) * (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (-1.0))*1.))))))) / 2.0)) 
    v["i362"] = 0.087000*np.tanh(((((((((1.570796) > (data["AMT_ANNUITY"]))*1.)) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.minimum(((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]>0, (-1.0*((data["AMT_INCOME_TOTAL"]))), data["AMT_ANNUITY"] ))), ((data["NAME_INCOME_TYPE_Student"])))))/2.0)))*1.)) * 2.0)) 
    v["i363"] = 0.099560*np.tanh(((((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) * 2.0)) * 2.0)) * ((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) > ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) + ((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) + ((2.05832409858703613)))) + ((2.05832052230834961)))/2.0)))/2.0)))*1.)))) * 2.0)) 
    v["i364"] = 0.099197*np.tanh((-1.0*((np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]<0, np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]<0, np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"]<0, (((data["NAME_INCOME_TYPE_Student"]) < (data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"]))*1.), (5.0) ), -2.0 ), data["CC_AMT_PAYMENT_CURRENT_MAX"] ))))) 
    v["i365"] = 0.089978*np.tanh(np.minimum((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MAX"]) < (data["CC_AMT_BALANCE_MAX"]))*1.))), ((np.where(data["CC_AMT_BALANCE_MEAN"]>0, data["CC_MONTHS_BALANCE_SUM"], (((((((((data["CC_AMT_BALANCE_MAX"]) < (data["CC_MONTHS_BALANCE_SUM"]))*1.)) < (data["CC_AMT_RECEIVABLE_PRINCIPAL_MAX"]))*1.)) < (data["CC_AMT_RECIVABLE_MAX"]))*1.) ))))) 
    v["i366"] = 0.099495*np.tanh(((((((((0.318310) > ((-1.0*((((data["CC_AMT_BALANCE_MAX"]) * 2.0))))))*1.)) < (((data["CC_NAME_CONTRACT_STATUS_Refused_SUM"]) + ((((np.maximum(((data["AMT_INCOME_TOTAL"])), ((data["CC_AMT_TOTAL_RECEIVABLE_MAX"])))) + (data["CC_AMT_DRAWINGS_OTHER_CURRENT_SUM"]))/2.0)))))*1.)) * 2.0)) 
    v["i367"] = 0.099650*np.tanh(((np.minimum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])), ((np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"]>0, 1.570796, ((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]) - ((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (((data["CC_AMT_PAYMENT_CURRENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"]))))*1.))) ))))) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"]))) 
    v["i368"] = 0.085001*np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((-1.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) * (np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, data["ORGANIZATION_TYPE_Industry__type_12"], (((data["CC_AMT_PAYMENT_CURRENT_VAR"]) > (((data["ORGANIZATION_TYPE_Industry__type_12"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.) )))) 
    v["i369"] = 0.081437*np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_MAX"] < -99998, 0.0, ((np.maximum(((((np.minimum(((0.0)), ((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])))) + ((((2.0) < (data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"]))*1.))))), ((((data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"]) * 2.0))))) * 2.0) )) 
    v["i370"] = 0.097400*np.tanh(((((data["ORGANIZATION_TYPE_Industry__type_12"]) * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0)))) + ((((((data["ORGANIZATION_TYPE_Trade__type_4"]) > (((((data["ORGANIZATION_TYPE_Industry__type_12"]) * (np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) / 2.0)))*1.)) / 2.0)))) 
    v["i371"] = 0.065150*np.tanh(np.minimum(((np.minimum((((((9.97522354125976562)) * (((data["ORGANIZATION_TYPE_Trade__type_4"]) * (data["AMT_ANNUITY"])))))), ((((data["ORGANIZATION_TYPE_Industry__type_12"]) * (data["REGION_POPULATION_RELATIVE"]))))))), ((((0.318310) * (np.tanh((data["REGION_POPULATION_RELATIVE"])))))))) 
    v["i372"] = 0.099700*np.tanh(np.minimum(((((data["ORGANIZATION_TYPE_Trade__type_4"]) * (-2.0)))), ((((((((-2.0) * 2.0)) * 2.0)) * (((data["ORGANIZATION_TYPE_Industry__type_12"]) + (((-2.0) - ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))))))) 
    v["i373"] = 0.089000*np.tanh((((np.where(np.where(data["AMT_INCOME_TOTAL"]<0, data["AMT_ANNUITY"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )<0, data["AMT_INCOME_TOTAL"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) > (np.maximum(((2.0)), ((((data["AMT_ANNUITY"]) * (3.0)))))))*1.)) 
    v["i374"] = 0.010549*np.tanh(((3.141593) * ((-1.0*(((((3.141593) < (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((data["ORGANIZATION_TYPE_Transport__type_1"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((3.0) / 2.0))))) )))*1.))))))) 
    v["i375"] = 0.070390*np.tanh(((data["FONDKAPREMONT_MODE_not_specified"]) * (((data["FONDKAPREMONT_MODE_not_specified"]) * ((((data["AMT_INCOME_TOTAL"]) + (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((((-1.0*((data["AMT_INCOME_TOTAL"])))) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))*1.)) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)))/2.0)))))) 
    v["i376"] = 0.096994*np.tanh((-1.0*((np.where(data["FONDKAPREMONT_MODE_reg_oper_spec_account"]>0, (((-1.0) + (data["AMT_ANNUITY"]))/2.0), ((data["NAME_INCOME_TYPE_Student"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NAME_INCOME_TYPE_Student"])))) ))))) 
    v["i377"] = 0.005040*np.tanh(((((data["ORGANIZATION_TYPE_Realtor"]) - (data["NAME_INCOME_TYPE_Student"]))) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.maximum(((((data["ORGANIZATION_TYPE_Realtor"]) - (data["NAME_INCOME_TYPE_Student"])))), ((data["ORGANIZATION_TYPE_Legal_Services"])))))))))) 
    v["i378"] = 0.099709*np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (1.570796)), data["AMT_ANNUITY"] )) > (np.where(data["AMT_ANNUITY"]>0, 3.141593, data["ORGANIZATION_TYPE_Trade__type_4"] )))*1.) )) 
    v["i379"] = 0.049996*np.tanh(np.where(data["AMT_INCOME_TOTAL"]<0, ((data["ORGANIZATION_TYPE_Legal_Services"]) * 2.0), ((data["ORGANIZATION_TYPE_Legal_Services"]) - (((np.where(data["AMT_ANNUITY"]<0, data["AMT_INCOME_TOTAL"], np.maximum(((data["ORGANIZATION_TYPE_Legal_Services"])), ((data["ORGANIZATION_TYPE_Trade__type_4"]))) )) * (data["AMT_ANNUITY"])))) )) 
    v["i380"] = 0.040001*np.tanh(np.minimum((((((((7.0)) * (data["ORGANIZATION_TYPE_Transport__type_1"]))) * (np.minimum((((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))), ((data["AMT_ANNUITY"]))))))), ((((((data["ORGANIZATION_TYPE_Transport__type_1"]) * (data["AMT_ANNUITY"]))) * (data["AMT_ANNUITY"])))))) 
    v["i381"] = 0.082000*np.tanh((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0))*1.)) / 2.0)) - ((((1.570796) < (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["ORGANIZATION_TYPE_Transport__type_1"]))/2.0)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)))*1.)))) * ((4.13047790527343750)))) 
    v["i382"] = 0.071302*np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.maximum(((data["AMT_ANNUITY"])), ((((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((3.56457448005676270)) / 2.0))))) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (np.tanh((data["AMT_ANNUITY"])))))))))))*1.)) 
    v["i383"] = 0.0*np.tanh(np.where(data["ORGANIZATION_TYPE_Industry__type_5"]<0, (((-1.0*((data["ORGANIZATION_TYPE_Industry__type_5"])))) + (np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0))*1.)>0, 3.141593, data["ORGANIZATION_TYPE_Realtor"] ))), (-1.0*((2.0))) )) 
    v["i384"] = 0.000002*np.tanh(((data["ORGANIZATION_TYPE_Realtor"]) - ((((((((((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["NAME_TYPE_SUITE_Group_of_people"]))/2.0)) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) + ((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["NAME_INCOME_TYPE_Student"]))/2.0)))/2.0)) + (data["ORGANIZATION_TYPE_Industry__type_13"]))/2.0)))) 
    v["i385"] = 0.099700*np.tanh((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((2.0)) + ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (((2.0) + (data["NAME_INCOME_TYPE_Maternity_leave"]))))) > ((0.11299613118171692)))*1.)))))*1.)) * 2.0)) * 2.0)) * 2.0)) 
    v["i386"] = 0.085100*np.tanh(((((((((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)) < (0.636620))*1.)) - (np.maximum(((data["NAME_INCOME_TYPE_Student"])), ((((((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)) < (0.318310))*1.))))))) / 2.0)) 
    v["i387"] = 0.074910*np.tanh(((((((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((((2.60535907745361328)) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))*1.)))/2.0)))/2.0) )) * 2.0)) * ((11.63154888153076172)))) * 2.0)) 
    v["i388"] = 0.099690*np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((2.0)))*1.)) - (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, data["CLOSED_MONTHS_BALANCE_MAX_MAX"], (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["CLOSED_MONTHS_BALANCE_MAX_MAX"] )))*1.) )))) 
    v["i389"] = 0.099990*np.tanh(np.where(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["CC_NAME_CONTRACT_STATUS_Active_MEAN"] )<0, (((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["CC_NAME_CONTRACT_STATUS_Active_MEAN"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) < (data["CC_NAME_CONTRACT_STATUS_Active_MEAN"]))*1.), (((data["CC_NAME_CONTRACT_STATUS_Active_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0) )) 
    v["i390"] = 0.099600*np.tanh(np.minimum((((((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) > (np.tanh((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) / 2.0)))))*1.)) + (((data["CC_AMT_PAYMENT_CURRENT_VAR"]) + ((-1.0*((data["CC_AMT_PAYMENT_CURRENT_MEAN"]))))))))), (((((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]) > (data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]))*1.))))) 
    v["i391"] = 0.097600*np.tanh((-1.0*(((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (((((((-1.0*((np.maximum((((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (0.636620))*1.))), ((data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"]))))))) < (data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]))*1.)) - (data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]))))*1.))))) 
    v["i392"] = 0.092000*np.tanh((((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]) > ((((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) + (np.where(((((-1.0*((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"])))) + (data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]))/2.0)>0, data["CC_AMT_PAYMENT_CURRENT_MEAN"], data["CC_AMT_PAYMENT_CURRENT_VAR"] )))/2.0)) / 2.0)))*1.)) 
    v["i393"] = 0.066595*np.tanh(np.where((((2.0) < (((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) * 2.0)))*1.)>0, (-1.0*((((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) * 2.0)))), np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"] < -99998, ((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"])), data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"] ) )) 
    v["i394"] = 0.100000*np.tanh((-1.0*(((((((((((((((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) / 2.0)) / 2.0)) > ((-1.0*((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])))))*1.)) + (0.636620))/2.0)) + (data["CC_AMT_PAYMENT_CURRENT_MAX"]))) < (data["CC_AMT_PAYMENT_CURRENT_MEAN"]))*1.))))) 
    v["i395"] = 0.022210*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((((-1.0*(((((data["ORGANIZATION_TYPE_Realtor"]) + (data["AMT_INCOME_TOTAL"]))/2.0))))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.), (((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (np.maximum((((-1.0*((data["ORGANIZATION_TYPE_Industry__type_13"]))))), ((data["AMT_INCOME_TOTAL"])))))*1.) )) 
    v["i396"] = 0.000300*np.tanh(np.where(data["ORGANIZATION_TYPE_Business_Entity_Type_1"]<0, (((((((np.tanh((data["ORGANIZATION_TYPE_Business_Entity_Type_1"]))) < ((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0)))/2.0)))*1.)) / 2.0)) / 2.0), ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_1"])) )) 
    v["i397"] = 0.085680*np.tanh((((data["ORGANIZATION_TYPE_Business_Entity_Type_1"]) + (np.where(data["ORGANIZATION_TYPE_Business_Entity_Type_1"]>0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], np.where((((((1.570796) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((-1.0*((0.318310)))))*1.)>0, -2.0, 0.318310 ) )))/2.0)) 
    v["i398"] = 0.099020*np.tanh(((np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0))*1.)) * 2.0))), (((((2.0) + (((data["FLAG_OWN_REALTY"]) * (((data["AMT_ANNUITY"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))/2.0))))) * 2.0)) 
    v["i399"] = 0.099771*np.tanh(((((((13.78084564208984375)) * (np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"]<0, np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), (((((data["AMT_ANNUITY"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)))), data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] )))) < (np.minimum(((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["FLAG_OWN_REALTY"])))))*1.)) 
    v["i400"] = 0.058997*np.tanh(np.where((((data["CC_AMT_PAYMENT_CURRENT_VAR"]) > (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]))*1.)>0, data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"], np.where((((((0.318310) * 2.0)) < (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]))*1.)>0, data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"], ((data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"]) - (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"])) ) )) 
    v["i401"] = 0.051605*np.tanh((((-1.0*((((np.where(data["CC_AMT_PAYMENT_CURRENT_MAX"] < -99998, ((data["CC_CNT_DRAWINGS_OTHER_CURRENT_VAR"]) - (data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])), ((data["CC_AMT_PAYMENT_CURRENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_VAR"])) )) * (data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])))))) * 2.0)) 
    v["i402"] = 0.071032*np.tanh(np.where(data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]<0, ((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) - (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]))) * (data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"])), (((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) < (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]))*1.)) - (data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])) )) 
    v["i403"] = 0.099596*np.tanh((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (np.where((((((data["NAME_FAMILY_STATUS_Civil_marriage"]) / 2.0)) > (((((-1.0*((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) + (data["FLAG_OWN_REALTY"]))/2.0)))*1.)>0, (9.0), ((((10.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.) )))*1.)) 
    v["i404"] = 0.009700*np.tanh((((((((1.570796) < ((-1.0*((np.where((((data["AMT_ANNUITY"]) > (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, 1.570796, data["NEW_ANNUITY_TO_INCOME_RATIO"] )))*1.)>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NEW_ANNUITY_TO_INCOME_RATIO"] ))))))*1.)) * 2.0)) * 2.0)) 
    v["i405"] = 0.012100*np.tanh(((np.where(data["ORGANIZATION_TYPE_Mobile"]>0, ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["AMT_ANNUITY"])), (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((3.0) + (np.minimum(((data["AMT_INCOME_TOTAL"])), ((data["ORGANIZATION_TYPE_Mobile"])))))))*1.) )) * ((6.0)))) 
    v["i406"] = 0.012404*np.tanh((-1.0*((((data["WALLSMATERIAL_MODE_Mixed"]) * (((data["WALLSMATERIAL_MODE_Mixed"]) * ((((-1.0) + (((data["WALLSMATERIAL_MODE_Mixed"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["AMT_ANNUITY"]))/2.0)))))/2.0))))))))) 
    v["i407"] = 0.097502*np.tanh(np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["AMT_ANNUITY"]))>0, (((((((((data["AMT_ANNUITY"]) > (3.141593))*1.)) * (data["AMT_ANNUITY"]))) - (data["ORGANIZATION_TYPE_Realtor"]))) * 2.0), data["ORGANIZATION_TYPE_Realtor"] )) 
    v["i408"] = 0.099970*np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((((np.maximum((((2.58687329292297363))), (((((((((1.570796) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * (3.141593))) * 2.0))))) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) * 2.0) )) 
    v["i409"] = 0.084550*np.tanh((-1.0*((np.where((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > ((9.0)))*1.)>0, (10.0), np.where((((data["AMT_INCOME_TOTAL"]) + (-2.0))/2.0)>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((data["ORGANIZATION_TYPE_Industry__type_13"]) > ((10.0)))*1.) ) ))))) 
    v["i410"] = 0.037702*np.tanh(((data["ORGANIZATION_TYPE_Realtor"]) + (((data["ORGANIZATION_TYPE_Realtor"]) + ((((np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["ORGANIZATION_TYPE_Realtor"]))*1.)>0, data["ORGANIZATION_TYPE_Realtor"], ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (1.0)) )) > (3.0))*1.)))))) 
    v["i411"] = 0.085500*np.tanh((((-1.0*(((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (np.where(((1.570796) + (data["AMT_ANNUITY"]))<0, ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (1.0)))), (8.41182804107666016) )))*1.))))) * 2.0)) 
    v["i412"] = 0.011900*np.tanh((((((((1.570796) < (((((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)))*1.)) * (((data["ORGANIZATION_TYPE_Mobile"]) - (((1.570796) * 2.0)))))) * 2.0)) 
    v["i413"] = 0.097005*np.tanh(((np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((data["ORGANIZATION_TYPE_Mobile"]))))), ((data["NAME_INCOME_TYPE_Student"])))) * (((np.where(-1.0>0, data["AMT_ANNUITY"], np.tanh((-1.0)) )) - (data["AMT_ANNUITY"]))))) 
    v["i414"] = 0.029970*np.tanh(((((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, -2.0, data["ORGANIZATION_TYPE_Trade__type_4"] )) - ((-1.0*((data["NAME_TYPE_SUITE_Group_of_people"])))))) * (data["ORGANIZATION_TYPE_Mobile"]))) + (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, data["NAME_TYPE_SUITE_Group_of_people"], data["ORGANIZATION_TYPE_Mobile"] )))) 
    v["i415"] = 0.098026*np.tanh(((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, ((((-1.0*((((data["AMT_INCOME_TOTAL"]) * (data["AMT_INCOME_TOTAL"])))))) > (np.where(data["AMT_INCOME_TOTAL"]>0, data["ORGANIZATION_TYPE_Trade__type_7"], data["ORGANIZATION_TYPE_Agriculture"] )))*1.), data["AMT_INCOME_TOTAL"] )) * (data["ORGANIZATION_TYPE_Trade__type_7"]))) 
    v["i416"] = 0.074000*np.tanh((-1.0*(((((3.141593) < ((((np.where(data["AMT_INCOME_TOTAL"]<0, ((3.141593) * 2.0), data["BURO_CNT_CREDIT_PROLONG_SUM"] )) + ((((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) + ((((3.52182722091674805)) * (data["BURO_CREDIT_TYPE_Car_loan_MEAN"]))))/2.0)))/2.0)))*1.))))) 
    v["i417"] = 0.029946*np.tanh(np.where(data["ORGANIZATION_TYPE_Mobile"]<0, ((((data["ORGANIZATION_TYPE_Agriculture"]) * ((((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (data["CODE_GENDER"]))))) + (data["ORGANIZATION_TYPE_Mobile"]))/2.0)))) - (data["NAME_INCOME_TYPE_Student"])), data["ORGANIZATION_TYPE_Mobile"] )) 
    v["i418"] = 0.090020*np.tanh(np.where(np.where(data["NEW_INC_PER_CHLD"]<0, data["AMT_INCOME_TOTAL"], data["ORGANIZATION_TYPE_Agriculture"] )<0, ((data["WALLSMATERIAL_MODE_Mixed"]) * ((-1.0*((data["ORGANIZATION_TYPE_Industry__type_11"]))))), np.where(data["ORGANIZATION_TYPE_Agriculture"]<0, data["AMT_INCOME_TOTAL"], (-1.0*((data["ORGANIZATION_TYPE_Agriculture"]))) ) )) 
    v["i419"] = 0.100000*np.tanh(((np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], ((np.where(((data["AMT_INCOME_TOTAL"]) - (data["NEW_INC_PER_CHLD"]))<0, data["NEW_INC_PER_CHLD"], (4.0) )) - (((2.0) / 2.0))) )) * ((6.28400230407714844)))) 
    v["i420"] = 0.099760*np.tanh(np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((((((data["NEW_INC_PER_CHLD"]) + ((((2.0) + (data["AMT_INCOME_TOTAL"]))/2.0)))) < (((((((data["AMT_INCOME_TOTAL"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.)) < (data["NEW_INC_PER_CHLD"]))*1.)))*1.))))) 
    v["i421"] = 0.092981*np.tanh((((((((data["CNT_CHILDREN"]) > ((((((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), ((data["ORGANIZATION_TYPE_Trade__type_4"])))) < ((((((data["CNT_CHILDREN"]) > (data["ORGANIZATION_TYPE_Trade__type_7"]))*1.)) / 2.0)))*1.)) * 2.0)) * 2.0)))*1.)) * 2.0)) * 2.0)) 
    v["i422"] = 0.099500*np.tanh(((np.minimum(((np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]<0, -2.0, data["NEW_INC_PER_CHLD"] ))), ((data["AMT_INCOME_TOTAL"])))) * ((((np.maximum(((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])), (((((0.636620) + (data["NEW_INC_PER_CHLD"]))/2.0))))) < (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))*1.)))) 
    v["i423"] = 0.001995*np.tanh((((0.0)) - (np.where(data["NEW_DOC_IND_KURT"]>0, data["ORGANIZATION_TYPE_Trade__type_4"], ((((((data["ORGANIZATION_TYPE_Trade__type_7"]) < (data["AMT_INCOME_TOTAL"]))*1.)) + ((((data["ORGANIZATION_TYPE_Trade__type_7"]) + ((((data["NEW_DOC_IND_KURT"]) + (data["AMT_INCOME_TOTAL"]))/2.0)))/2.0)))/2.0) )))) 
    v["i424"] = 0.096014*np.tanh((((((-1.0*((np.maximum(((data["NAME_EDUCATION_TYPE_Academic_degree"])), (((-1.0*((data["NAME_TYPE_SUITE_Group_of_people"])))))))))) * ((-1.0*((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])))))) - (((data["ORGANIZATION_TYPE_Trade__type_4"]) + ((((data["ORGANIZATION_TYPE_Trade__type_7"]) > (0.636620))*1.)))))) 
    v["i425"] = 0.097301*np.tanh(((data["FONDKAPREMONT_MODE_not_specified"]) * (((((data["NEW_DOC_IND_KURT"]) * ((((data["NEW_DOC_IND_KURT"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) * (np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((data["AMT_INCOME_TOTAL"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.))))))))) 
    v["i426"] = 0.089008*np.tanh(np.where((((data["ORGANIZATION_TYPE_Trade__type_7"]) + (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]))/2.0)>0, data["FLAG_DOCUMENT_3"], ((data["ORGANIZATION_TYPE_Security_Ministries"]) * (np.where(data["NAME_TYPE_SUITE_Group_of_people"]>0, data["FLAG_DOCUMENT_3"], ((data["ORGANIZATION_TYPE_Security_Ministries"]) * (data["NEW_INC_PER_CHLD"])) ))) )) 
    v["i427"] = 0.099750*np.tanh(np.where(((data["EXT_SOURCE_2"]) - (data["NAME_INCOME_TYPE_Student"]))<0, data["AMT_INCOME_TOTAL"], (((((-1.0*((data["EXT_SOURCE_2"])))) * (((data["EXT_SOURCE_2"]) - ((((data["EXT_SOURCE_2"]) < (1.0))*1.)))))) * 2.0) )) 
    v["i428"] = 0.099890*np.tanh((((((np.minimum(((data["NEW_EMPLOY_TO_BIRTH_RATIO"])), ((data["DAYS_EMPLOYED"])))) > (np.where(((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) * (data["DAYS_EMPLOYED"]))<0, ((data["NAME_INCOME_TYPE_Student"]) * 2.0), np.tanh((np.tanh((data["NEW_EMPLOY_TO_BIRTH_RATIO"])))) )))*1.)) * 2.0)) 
    v["i429"] = 0.091597*np.tanh(((((((((data["NEW_CAR_TO_EMPLOY_RATIO"]) - (data["DAYS_EMPLOYED"]))) > ((((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) > ((((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) > (0.636620))*1.)))*1.)))*1.)) > ((((1.570796) < (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))*1.)))*1.)) 
    v["i430"] = 0.090401*np.tanh(np.where(data["DAYS_EMPLOYED"] < -99998, ((3.141593) - (3.0)), np.where(data["FLAG_OWN_CAR"]<0, (((3.141593) < (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))*1.), ((3.0) * (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) ) )) 
    v["i431"] = 0.099920*np.tanh(((((((data["REFUSED_DAYS_DECISION_MEAN"]) * (np.where(data["REFUSED_CNT_PAYMENT_SUM"]<0, 0.0, ((np.where(data["REFUSED_AMT_APPLICATION_MEAN"]>0, data["REFUSED_CNT_PAYMENT_SUM"], 0.0 )) - (3.0)) )))) * (data["REFUSED_AMT_APPLICATION_MEAN"]))) * 2.0)) 
    v["i432"] = 0.091453*np.tanh((((((2.0) < (np.where(data["NAME_INCOME_TYPE_State_servant"]<0, (-1.0*((((data["EXT_SOURCE_2"]) - ((-1.0*((0.318310)))))))), data["REGION_POPULATION_RELATIVE"] )))*1.)) * 2.0)) 
    v["i433"] = 0.095979*np.tanh(np.where(((data["NAME_FAMILY_STATUS_Civil_marriage"]) - (((data["NEW_EXT_SOURCES_MEAN"]) + ((2.0)))))<0, (((np.where(data["NEW_EXT_SOURCES_MEAN"]<0, data["NEW_EXT_SOURCES_MEAN"], data["NAME_FAMILY_STATUS_Civil_marriage"] )) > (data["ORGANIZATION_TYPE_Military"]))*1.), data["NAME_INCOME_TYPE_State_servant"] )) 
    v["i434"] = 0.054620*np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, ((np.tanh((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))) * (np.minimum(((data["REGION_RATING_CLIENT"])), ((0.318310))))), np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((((data["REGION_RATING_CLIENT"]) + (data["REGION_POPULATION_RELATIVE"]))))) )) 
    v["i435"] = 0.011396*np.tanh(((((((np.maximum(((data["CC_AMT_RECIVABLE_MEAN"])), ((data["CC_AMT_BALANCE_MEAN"])))) - (data["CC_AMT_INST_MIN_REGULARITY_MEAN"]))) * 2.0)) * 2.0)) 
    v["i436"] = 0.090200*np.tanh((((((((data["CC_AMT_RECIVABLE_MEAN"]) > ((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) > (data["CC_AMT_TOTAL_RECEIVABLE_MEAN"]))*1.)) + (((((0.10667683929204941)) < (data["CC_AMT_TOTAL_RECEIVABLE_MEAN"]))*1.)))))*1.)) * 2.0)) * 2.0)) 
    v["i437"] = 0.088430*np.tanh(np.where(np.where(data["NAME_TYPE_SUITE_Group_of_people"]<0, data["ORGANIZATION_TYPE_Trade__type_7"], data["NEW_EXT_SOURCES_MEAN"] )<0, (((data["NEW_EXT_SOURCES_MEAN"]) > (1.570796))*1.), np.where(data["NEW_EXT_SOURCES_MEAN"]<0, data["ORGANIZATION_TYPE_Trade__type_7"], -2.0 ) )) 
    v["i438"] = 0.013965*np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"]>0, ((np.tanh((np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) / 2.0), (-1.0*((np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)) / 2.0)) / 2.0))))))) )) 
    v["i439"] = 0.097690*np.tanh((-1.0*(((((-1.0) > (np.maximum((((((((data["NEW_EXT_SOURCES_MEAN"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (((-1.0) * 2.0)))/2.0))), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) + (0.636620)))))))*1.))))) 
    v["i440"] = 0.099999*np.tanh((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((3.86333322525024414)))*1.)) * (data["AMT_CREDIT"]))) - (((((3.86333322525024414)) < (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((data["AMT_CREDIT"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))))*1.)))) 
    v["i441"] = 0.099940*np.tanh(np.maximum((((((np.tanh((1.570796))) < (((data["NEW_CAR_TO_EMPLOY_RATIO"]) - ((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0)))))*1.))), (((((((data["AMT_CREDIT"]) < (data["NEW_EXT_SOURCES_MEAN"]))*1.)) * (data["AMT_CREDIT"])))))) 
    v["i442"] = 0.020019*np.tanh((((np.where(((-1.0) + (data["AMT_CREDIT"]))<0, data["OWN_CAR_AGE"], ((np.where(data["NEW_EXT_SOURCES_MEAN"]<0, ((-1.0) + (data["AMT_CREDIT"])), -1.0 )) * 2.0) )) > ((1.32165348529815674)))*1.)) 
    v["i443"] = 0.090000*np.tanh(((((np.where((-1.0*((((0.636620) + (((data["AMT_CREDIT"]) / 2.0))))))<0, ((data["ORGANIZATION_TYPE_Trade__type_5"]) * (((0.636620) + (data["FLAG_OWN_CAR"])))), data["NEW_CAR_TO_BIRTH_RATIO"] )) * 2.0)) * 2.0)) 
    v["i444"] = 0.069500*np.tanh((((((((((3.0) - (data["AMT_CREDIT"]))) < ((((np.where(data["NEW_EXT_SOURCES_MEAN"]>0, 0.318310, ((3.0) - (0.318310)) )) + (data["BURO_STATUS_5_MEAN_MEAN"]))/2.0)))*1.)) * 2.0)) * 2.0)) 
    v["i445"] = 0.099749*np.tanh(np.where(((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((data["AMT_CREDIT"]) < (data["NEW_EXT_SOURCES_MEAN"]))*1.)))>0, data["NEW_EXT_SOURCES_MEAN"], ((data["AMT_CREDIT"]) * (np.minimum(((data["NAME_EDUCATION_TYPE_Lower_secondary"])), ((((data["NAME_INCOME_TYPE_Student"]) - (data["NAME_EDUCATION_TYPE_Lower_secondary"]))))))) )) 
    v["i446"] = 0.033999*np.tanh((-1.0*(((((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["NEW_CAR_TO_EMPLOY_RATIO"]))*1.)) * (np.maximum(((np.where(data["OWN_CAR_AGE"]<0, ((data["NEW_CAR_TO_BIRTH_RATIO"]) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), (-1.0*((data["NEW_CREDIT_TO_INCOME_RATIO"]))) ))), ((data["NEW_EXT_SOURCES_MEAN"]))))))))) 
    v["i447"] = 0.099649*np.tanh(((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (-2.0))) + (((3.0) / 2.0)))) * ((((data["NEW_EXT_SOURCES_MEAN"]) > ((((2.36968207359313965)) / 2.0)))*1.)))) 
    v["i448"] = 0.099899*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]<0, data["ACTIVE_DAYS_CREDIT_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_SUM"] ), ((((((3.0) < (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * 2.0)))*1.)) > ((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (2.0))*1.)))*1.) )) 
    v["i449"] = 0.049990*np.tanh(((((((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) * 2.0)) * 2.0)) * (np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]<0, (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) < (data["ACTIVE_AMT_CREDIT_SUM_SUM"]))*1.), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) - ((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.))) )))) 
    v["i450"] = 0.099984*np.tanh((((data["NEW_EXT_SOURCES_MEAN"]) > (((((((3.0) + (data["ORGANIZATION_TYPE_Industry__type_5"]))) + (data["ORGANIZATION_TYPE_Trade__type_4"]))) - (np.where(((data["NEW_EXT_SOURCES_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))>0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["ORGANIZATION_TYPE_Industry__type_1"] )))))*1.)) 
    v["i451"] = 0.097200*np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"]<0, data["ORGANIZATION_TYPE_Advertising"], np.where(((data["AMT_ANNUITY"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))<0, -1.0, (((data["NEW_EXT_SOURCES_MEAN"]) < ((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["AMT_ANNUITY"]))/2.0)))*1.) ) )) 
    v["i452"] = 0.099052*np.tanh((-1.0*((np.where(np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"]>0, data["NEW_CAR_TO_EMPLOY_RATIO"], data["NEW_EXT_SOURCES_MEAN"] ) < -99998, 1.570796, np.where(data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]<0, np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"]>0, -2.0, data["ORGANIZATION_TYPE_Industry__type_5"] ), 3.141593 ) ))))) 
    v["i453"] = 0.056500*np.tanh((((((((data["NEW_CAR_TO_BIRTH_RATIO"]) > ((((((2.0) * ((-1.0*((data["ORGANIZATION_TYPE_Trade__type_4"])))))) < ((-1.0*(((((data["OWN_CAR_AGE"]) + (data["NEW_CAR_TO_EMPLOY_RATIO"]))/2.0))))))*1.)))*1.)) * 2.0)) * 2.0)) 
    v["i454"] = 0.099945*np.tanh(np.where(((data["FLAG_EMP_PHONE"]) + (data["NEW_CAR_TO_EMPLOY_RATIO"]))<0, (((1.570796) < (data["OWN_CAR_AGE"]))*1.), np.where((((data["FLAG_EMP_PHONE"]) > (data["NEW_CAR_TO_BIRTH_RATIO"]))*1.)>0, data["NEW_CAR_TO_EMPLOY_RATIO"], data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] ) )) 
    v["i455"] = 0.098400*np.tanh(((((((((0.636620) > (data["OWN_CAR_AGE"]))*1.)) > (np.tanh(((((data["DAYS_EMPLOYED"]) + (data["DAYS_EMPLOYED"]))/2.0)))))*1.)) * ((-1.0*(((((data["DAYS_EMPLOYED"]) > ((-1.0*((data["OWN_CAR_AGE"])))))*1.))))))) 
    v["i456"] = 0.028998*np.tanh((-1.0*((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (np.where(data["ORGANIZATION_TYPE_Trade__type_4"]<0, ((data["NAME_INCOME_TYPE_Student"]) + (((np.minimum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((data["ORGANIZATION_TYPE_Security_Ministries"])))) + (data["NAME_INCOME_TYPE_Student"])))), 2.0 ))))))) 
    v["i457"] = 0.069700*np.tanh((((data["ORGANIZATION_TYPE_Transport__type_3"]) > (np.where(((((((data["AMT_ANNUITY"]) + ((((data["ORGANIZATION_TYPE_Realtor"]) + (data["AMT_ANNUITY"]))/2.0)))/2.0)) + ((((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)))/2.0)<0, data["ORGANIZATION_TYPE_Transport__type_3"], data["AMT_ANNUITY"] )))*1.)) 
    v["i458"] = 0.099350*np.tanh(np.maximum((((-1.0*((data["NAME_INCOME_TYPE_Student"]))))), ((np.where(np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, 0.318310, data["NEW_CAR_TO_EMPLOY_RATIO"] )>0, data["OWN_CAR_AGE"], (-1.0*((((data["OWN_CAR_AGE"]) * 2.0)))) ))))) 
    v["i459"] = 0.074200*np.tanh((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (np.where(data["NEW_INC_PER_CHLD"]<0, (8.19191551208496094), ((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) + (np.where((((data["NEW_INC_PER_CHLD"]) + (-1.0))/2.0)<0, 2.0, data["NAME_INCOME_TYPE_Student"] ))) )))*1.)) 
    v["i460"] = 0.093970*np.tanh((((-1.0*(((((3.0) < (np.where(np.where(data["AMT_INCOME_TOTAL"]>0, data["ORGANIZATION_TYPE_Business_Entity_Type_3"], 0.636620 )>0, data["NEW_CREDIT_TO_INCOME_RATIO"], ((data["AMT_CREDIT"]) - (data["ORGANIZATION_TYPE_Business_Entity_Type_3"])) )))*1.))))) * (3.141593))) 
    v["i461"] = 0.080000*np.tanh((((-2.0) > ((((((np.where(data["AMT_ANNUITY"]>0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((-1.0*((data["AMT_INCOME_TOTAL"])))) * 2.0))), (-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) )) * 2.0)) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"]))/2.0)))*1.)) 
    v["i462"] = 0.099397*np.tanh((-1.0*(((((((((((3.141593) - (((data["REGION_RATING_CLIENT"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) < ((((((3.141593) - (data["REGION_RATING_CLIENT"]))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))*1.)) * 2.0)) * 2.0))))) 
    v["i463"] = 0.092303*np.tanh(((((np.where((((((data["NAME_INCOME_TYPE_Student"]) * (data["ORGANIZATION_TYPE_Industry__type_13"]))) < (data["ORGANIZATION_TYPE_Security_Ministries"]))*1.)>0, data["FONDKAPREMONT_MODE_reg_oper_account"], ((data["FONDKAPREMONT_MODE_reg_oper_account"]) * ((-1.0*((data["ORGANIZATION_TYPE_Industry__type_13"]))))) )) * 2.0)) * 2.0)) 
    v["i464"] = 0.018040*np.tanh((((((np.where(data["FLOORSMAX_MODE"]<0, data["FLOORSMAX_MODE"], data["NONLIVINGAREA_MEDI"] )) < (((1.0) - ((((2.44985151290893555)) - (data["LIVINGAPARTMENTS_AVG"]))))))*1.)) - ((((data["FLOORSMAX_MODE"]) > (1.0))*1.)))) 
    v["i465"] = 0.070000*np.tanh((((((data["LIVINGAPARTMENTS_MODE"]) < (np.where(data["LIVINGAPARTMENTS_MODE"]>0, 0.318310, data["LIVINGAPARTMENTS_AVG"] )))*1.)) * (((((-1.0) + (((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * 2.0)))) * 2.0)))) 
    v["i466"] = 0.094990*np.tanh(((((((((data["CC_NAME_CONTRACT_STATUS_Refused_MAX"]) + ((((data["CC_CNT_DRAWINGS_CURRENT_MEAN"]) > (((((np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0)) / 2.0)))*1.)))/2.0)) > ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["AMT_ANNUITY"]))*1.)))*1.)) * 2.0)) 
    v["i467"] = 0.069960*np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, (((data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]) < (data["CC_AMT_PAYMENT_CURRENT_MEAN"]))*1.), (-1.0*(((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (np.where(data["CC_AMT_PAYMENT_CURRENT_MAX"]<0, (((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (0.636620))*1.), 0.318310 )))*1.)))) )) 
    v["i468"] = 0.080521*np.tanh(np.tanh((np.maximum(((np.where(data["OCCUPATION_TYPE_Cooking_staff"]>0, data["NAME_EDUCATION_TYPE_Higher_education"], data["ACTIVE_DAYS_CREDIT_VAR"] ))), (((((((((data["NAME_EDUCATION_TYPE_Higher_education"]) + ((((data["OCCUPATION_TYPE_Cooking_staff"]) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.)))/2.0)) * (data["AMT_ANNUITY"]))) / 2.0))))))) 
    v["i469"] = 0.094998*np.tanh(((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) * (((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) + (np.maximum(((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])), (((((data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]) > (1.0))*1.))))))) * (((data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))))))) 
    v["i470"] = 0.094799*np.tanh(((np.minimum(((np.where(data["ACTIVE_DAYS_CREDIT_MEAN"] < -99998, 0.0, (((((((data["ACTIVE_DAYS_CREDIT_MEAN"]) < (-1.0))*1.)) * 2.0)) + (data["ACTIVE_DAYS_CREDIT_MEAN"])) ))), ((((((0.0)) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.))))) / 2.0)) 
    v["i471"] = 0.095500*np.tanh((((((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) < (data["ACTIVE_AMT_CREDIT_SUM_SUM"]))*1.)) * (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * (((np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], 0.636620 )) * (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))) * 2.0)) * 2.0)) 
    v["i472"] = 0.085900*np.tanh((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)) - (np.where(((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (data["AMT_INCOME_TOTAL"]))<0, ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["AMT_INCOME_TOTAL"])), (-1.0*((data["NEW_ANNUITY_TO_INCOME_RATIO"]))) )))) > (3.141593))*1.)) 
    v["i473"] = 0.099010*np.tanh((((((data["AMT_INCOME_TOTAL"]) > (np.where((((3.0) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)>0, ((data["AMT_INCOME_TOTAL"]) * (data["AMT_INCOME_TOTAL"])), ((3.0) / 2.0) )))*1.)) * (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))) 
    v["i474"] = 0.020201*np.tanh((((((-1.0*((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) - (data["OCCUPATION_TYPE_Cooking_staff"]))) - (np.tanh((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) / 2.0))))))))) * (data["ORGANIZATION_TYPE_University"]))) * (data["AMT_ANNUITY"]))) 
    v["i475"] = 0.099490*np.tanh((((((-1.0) > (((((data["AMT_ANNUITY"]) + (np.where(data["AMT_INCOME_TOTAL"]<0, np.where(data["AMT_ANNUITY"]<0, data["AMT_INCOME_TOTAL"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0) ), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )))) / 2.0)))*1.)) * 2.0)) 
    v["i476"] = 0.097001*np.tanh(np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] < -99998, 0.318310, (-1.0*((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (((1.0) < (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))*1.), ((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) - ((((-1.0*((1.0)))) / 2.0))) )))) )) 
    v["i477"] = 0.076000*np.tanh((((np.tanh((1.0))) < (np.where(data["CC_AMT_TOTAL_RECEIVABLE_MAX"]>0, data["CC_AMT_TOTAL_RECEIVABLE_MAX"], ((((((data["CC_COUNT"]) > (data["CC_NAME_CONTRACT_STATUS_Active_SUM"]))*1.)) < ((-1.0*((((data["CC_AMT_TOTAL_RECEIVABLE_MAX"]) * (data["CC_NAME_CONTRACT_STATUS_Active_SUM"])))))))*1.) )))*1.)) 
    v["i478"] = 0.058999*np.tanh((((np.where(data["APPROVED_RATE_DOWN_PAYMENT_MAX"] < -99998, data["APPROVED_RATE_DOWN_PAYMENT_MAX"], np.where(data["PREV_RATE_DOWN_PAYMENT_MAX"]>0, data["APPROVED_RATE_DOWN_PAYMENT_MAX"], data["PREV_RATE_DOWN_PAYMENT_MEAN"] ) )) > (np.where(data["PREV_RATE_DOWN_PAYMENT_MEAN"]>0, 1.570796, np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])) )))*1.)) 
    v["i479"] = 0.098290*np.tanh((((((data["REFUSED_APP_CREDIT_PERC_MAX"]) * ((((data["REFUSED_APP_CREDIT_PERC_MEAN"]) < ((((data["REFUSED_APP_CREDIT_PERC_MEAN"]) < (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))*1.)))*1.)))) > ((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) / 2.0)))*1.)))*1.)) 
    v["i480"] = 0.100000*np.tanh(((np.minimum(((((1.0) - (data["DAYS_BIRTH"])))), (((((((data["DAYS_BIRTH"]) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) < ((((data["DAYS_BIRTH"]) < (data["BURO_DAYS_CREDIT_VAR"]))*1.)))*1.))))) * (data["NAME_FAMILY_STATUS_Single___not_married"]))) 
    v["i481"] = 0.063000*np.tanh(np.where(data["NAME_FAMILY_STATUS_Single___not_married"]<0, ((np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]>0, np.minimum(((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])), ((data["FLAG_EMP_PHONE"]))), (((data["NAME_FAMILY_STATUS_Single___not_married"]) > (data["FLAG_EMP_PHONE"]))*1.) )) / 2.0), ((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) * 2.0) )) 
    v["i482"] = 0.098001*np.tanh((((((np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"] < -99998, data["NAME_INCOME_TYPE_Student"], data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] )) > (((3.0) - (np.where(data["BURO_DAYS_CREDIT_VAR"] < -99998, 3.0, data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] )))))*1.)) * (data["BURO_DAYS_CREDIT_VAR"]))) 
    v["i483"] = 0.099230*np.tanh((((7.0)) * ((((data["BURO_DAYS_CREDIT_VAR"]) > (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]<0, np.where(data["NAME_FAMILY_STATUS_Single___not_married"]<0, np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_FAMILY_STATUS_Single___not_married"], (7.0) ), data["NAME_FAMILY_STATUS_Single___not_married"] ), (2.46100592613220215) )))*1.)))) 
    v["i484"] = 0.099970*np.tanh((((((1.570796) < (data["BURO_DAYS_CREDIT_VAR"]))*1.)) * (np.where(np.where(data["AMT_INCOME_TOTAL"]>0, data["EXT_SOURCE_3"], data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] )>0, np.maximum(((data["EXT_SOURCE_3"])), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))), (-1.0*((1.570796))) )))) 
    v["i485"] = 0.027810*np.tanh((-1.0*((np.where(data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]<0, np.where(data["BURO_DAYS_CREDIT_VAR"]<0, ((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) < (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)) > ((((data["BURO_DAYS_CREDIT_VAR"]) > (-2.0))*1.)))*1.), data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"] ), -2.0 ))))) 
    v["i486"] = 0.099950*np.tanh((((((3.0) - ((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * 2.0)) + (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) + (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0)))))/2.0)))) < (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]<0, data["AMT_REQ_CREDIT_BUREAU_DAY"], data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] )))*1.)) 
    v["i487"] = 0.099950*np.tanh((((((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (0.636620))) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)) * 2.0)) * 2.0)) * (((((data["DAYS_BIRTH"]) + (data["DAYS_BIRTH"]))) + (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))))) 
    v["i488"] = 0.069600*np.tanh(((((((1.0)) < (((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) * 2.0)))*1.)) * (np.where(data["EXT_SOURCE_3"]<0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, -2.0, np.maximum(((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])), (((3.15189313888549805)))) ) )))) 
    v["i489"] = 0.098019*np.tanh((-1.0*(((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > ((-1.0*((np.where(data["EXT_SOURCE_3"] < -99998, data["FLAG_EMP_PHONE"], (((data["NAME_FAMILY_STATUS_Single___not_married"]) + ((-1.0*((((data["EXT_SOURCE_3"]) / 2.0))))))/2.0) ))))))*1.))))) 
    v["i490"] = 0.057200*np.tanh((((((((((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) < (1.570796))*1.)) + (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))/2.0)) > (0.318310))*1.)) + (0.0))/2.0)) > ((((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) < (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)))*1.)) 
    v["i491"] = 0.092970*np.tanh((((np.where(((data["NEW_INC_PER_CHLD"]) * (((((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (data["NEW_INC_PER_CHLD"]))*1.)) < (data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))*1.)))>0, (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (0.318310))*1.), (8.0) )) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))*1.)) 
    v["i492"] = 0.056000*np.tanh((((((((((1.570796) < (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)) * (((((data["EXT_SOURCE_3"]) * 2.0)) * 2.0)))) + (((((((data["EXT_SOURCE_3"]) < (1.570796))*1.)) < (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)))) * 2.0)) 
    v["i493"] = 0.098800*np.tanh((-1.0*((np.minimum((((((data["FLOORSMAX_AVG"]) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)))*1.))), ((((((((data["FLOORSMAX_MODE"]) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (0.636620))*1.)))*1.)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))))))) 
    v["i494"] = 0.099420*np.tanh(((((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) + (data["LIVINGAREA_MODE"]))/2.0)) > (np.where(data["ELEVATORS_AVG"]>0, np.where(data["REGION_POPULATION_RELATIVE"]>0, ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) * (data["ELEVATORS_AVG"])), 3.0 ), 0.636620 )))*1.)) 
    v["i495"] = 0.099459*np.tanh(((data["DAYS_BIRTH"]) * ((-1.0*((np.where(data["NEW_SOURCES_PROD"]<0, ((data["DAYS_BIRTH"]) * ((-1.0*(((((data["DAYS_BIRTH"]) < (data["NEW_SOURCES_PROD"]))*1.)))))), np.maximum(((data["NEW_SOURCES_PROD"])), ((data["DAYS_BIRTH"]))) ))))))) 
    v["i496"] = 0.021039*np.tanh(np.where(data["NEW_SOURCES_PROD"]<0, data["OCCUPATION_TYPE_Secretaries"], (-1.0*(((((((data["NEW_SOURCES_PROD"]) > ((((np.tanh((data["NEW_SOURCES_PROD"]))) > ((((data["NEW_SOURCES_PROD"]) + (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0)))/2.0)))*1.)))*1.)) * 2.0)))) )) 
    v["i497"] = 0.099830*np.tanh((((data["OCCUPATION_TYPE_Laborers"]) < (((((np.minimum(((data["NAME_INCOME_TYPE_Student"])), ((((((data["NEW_SOURCES_PROD"]) * 2.0)) - (data["EXT_SOURCE_1"])))))) * 2.0)) - (((data["NEW_SOURCES_PROD"]) / 2.0)))))*1.)) 
    v["i498"] = 0.038590*np.tanh(((np.where(data["ORGANIZATION_TYPE_Military"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0*(((((((np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((data["NEW_SOURCES_PROD"])))) + (np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (2.0))*1.)))) )) * 2.0)) 
    v["i499"] = 0.046598*np.tanh(((((((data["EXT_SOURCE_1"]) < (np.minimum((((((data["OCCUPATION_TYPE_Laborers"]) < (data["NEW_SOURCES_PROD"]))*1.))), ((data["NEW_SOURCES_PROD"])))))*1.)) + ((-1.0*(((((-2.0) < (np.minimum(((data["EXT_SOURCE_1"])), ((data["NEW_SOURCES_PROD"])))))*1.))))))/2.0)) 
    v["i500"] = 0.099900*np.tanh((((((data["NEW_SOURCES_PROD"]) > (np.where(data["NEW_SOURCES_PROD"]<0, data["NAME_INCOME_TYPE_Student"], np.maximum((((3.0))), ((data["AMT_ANNUITY"]))) )))*1.)) * (np.where(data["AMT_ANNUITY"]>0, (4.22247409820556641), data["NEW_SOURCES_PROD"] )))) 
    v["i501"] = 0.095040*np.tanh((((((((3.0) < (np.maximum(((data["NEW_SOURCES_PROD"])), ((np.where(np.maximum(((data["AMT_ANNUITY"])), ((data["OCCUPATION_TYPE_Laborers"])))<0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), data["AMT_INCOME_TOTAL"] ))))))*1.)) * 2.0)) * 2.0)) 
    v["i502"] = 0.001000*np.tanh((((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) > ((3.0)))*1.)) * (((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) * ((((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) > (3.141593))*1.)) + (((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) * 2.0)) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))) 
    v["i503"] = 0.099960*np.tanh(((np.where((((data["EXT_SOURCE_1"]) + ((((0.318310) + (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))/2.0)))/2.0)<0, 0.0, (-1.0*(((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (np.maximum(((data["NEW_SOURCES_PROD"])), ((data["NAME_INCOME_TYPE_Student"])))))*1.)))) )) * 2.0)) 
    v["i504"] = 0.000009*np.tanh(((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) - ((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) > (np.maximum(((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])), ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))))*1.)))) * ((((data["NEW_SOURCES_PROD"]) > ((((np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))*1.)))*1.)))) 
    v["i505"] = 0.014603*np.tanh((((((np.tanh((data["NEW_SOURCES_PROD"]))) > ((((data["PREV_AMT_DOWN_PAYMENT_MAX"]) > (np.tanh((np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))))))*1.)))*1.)) * (((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) + (-1.0))) + (-1.0))))) 
    v["i506"] = 0.064800*np.tanh(((((((data["NEW_SOURCES_PROD"]) < (((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) * 2.0)))*1.)) < ((((((((((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) < (((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) / 2.0)))*1.)) > (data["NEW_SOURCES_PROD"]))*1.)) / 2.0)) + (data["PREV_AMT_DOWN_PAYMENT_MAX"]))/2.0)))*1.)) 
    v["i507"] = 0.099096*np.tanh((((((np.where(data["NEW_SOURCES_PROD"] < -99998, data["AMT_ANNUITY"], data["NEW_SOURCES_PROD"] )) < (data["EXT_SOURCE_1"]))*1.)) * (np.where(data["NEW_SOURCES_PROD"]>0, data["AMT_ANNUITY"], np.minimum(((data["EXT_SOURCE_1"])), (((-1.0*((data["NEW_SOURCES_PROD"])))))) )))) 
    v["i508"] = 0.052010*np.tanh((-1.0*(((((((np.where(data["EXT_SOURCE_1"] < -99998, 3.141593, ((data["EXT_SOURCE_1"]) * (((((data["EXT_SOURCE_1"]) - (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))) - (data["NAME_INCOME_TYPE_Student"])))) )) < (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))*1.)) * 2.0))))) 
    v["i509"] = 0.099609*np.tanh((-1.0*(((((data["NEW_SOURCES_PROD"]) > (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, (((data["EXT_SOURCE_1"]) > (np.tanh((0.636620))))*1.), np.where(np.tanh((data["EXT_SOURCE_1"]))<0, 0.318310, data["AMT_ANNUITY"] ) )))*1.))))) 
    v["i510"] = 0.041999*np.tanh((((-1.0*(((((np.maximum((((((2.0) + (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))/2.0))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["ORGANIZATION_TYPE_Emergency"]))/2.0))))))) > (3.141593))*1.))))) * 2.0)) 
    v["i511"] = 0.092000*np.tanh(((((data["OCCUPATION_TYPE_Laborers"]) * 2.0)) * (((((-1.0*((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))) < (np.minimum(((((((((data["OCCUPATION_TYPE_Laborers"]) > (data["AMT_ANNUITY"]))*1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0))), ((data["AMT_ANNUITY"])))))*1.))))
    return v

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('input/application_train.csv', nrows= num_rows)
    test = pd.read_csv('input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test)))
    df = df.append(test).reset_index()
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    df = df.set_index("SK_ID_CURR")
    del test
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('input/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('input/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('input/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def prepare_gp2_features():
    debug = None
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left')
        del cc
        gc.collect()

    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    # for c in feats:
    #     print(c)
    #     ss = StandardScaler()
    #     df.loc[~np.isfinite(df[c]),c] = np.nan
    #     df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
    #     df[c].fillna(-99999.,inplace=True)

    train = df[df['TARGET'].notnull()]
    test = df[df['TARGET'].isnull()]
    print(train.shape)
    print(test.shape)

    train.columns = train.columns.str.replace('[^A-Za-z0-9_]', '_')
    test.columns = test.columns.str.replace('[^A-Za-z0-9_]', '_')
    feats = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
        
    floattypes = []
    inttypes = []
    stringtypes = []
    for c in test.columns:
        if(train[c].dtype=='object'):
            train[c] = train[c].astype('str')
            stringtypes.append(c)
        elif(train[c].dtype=='int64'):
            train[c] = train[c].astype('int32')
            inttypes.append(c)
        else:
            train[c] = train[c].astype('float32')
            floattypes.append(c)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in stringtypes:
        train['te_'+col] = 0.
        test['te_'+col] = 0.
        SMOOTHING = test[~test[col].isin(train[col])].shape[0]/test.shape[0]

        for f, (vis_index, blind_index) in enumerate(kf.split(train)):
            _, train.loc[blind_index, 'te_'+col] = target_encode(train.loc[vis_index, col], 
                                                                train.loc[blind_index, col], 
                                                                target=train.loc[vis_index,'TARGET'], 
                                                                min_samples_leaf=100,
                                                                smoothing=SMOOTHING,
                                                                noise_level=0.0)
            _, x = target_encode(train.loc[vis_index, col], 
                                              test[col], 
                                              target=train.loc[vis_index,'TARGET'], 
                                              min_samples_leaf=100,
                                              smoothing=SMOOTHING,
                                              noise_level=0.0)
            test['te_'+col] += (.2*x)
            
    ntrainrows = train.shape[0]
    del test["TARGET"]
    test.insert(1,'TARGET',-1)
    alldata = pd.concat([train,test])
    # del train ,test
    gc.collect()
    
    alldata['nans'] = alldata.isnull().sum(axis=1)
    
    for col in inttypes[1:]:
        x = alldata[col].value_counts().reset_index(drop=False)
        x.columns = [col,'cnt_'+col]
        x['cnt_'+col]/=alldata.shape[0]
        alldata = alldata.merge(x,on=col,how='left')
        
    features = list(set(alldata.columns).difference(['SK_ID_CURR','TARGET']))
    alldata[features] = alldata[features].astype('float32')
    for c in features:
        print(c)
        ss = StandardScaler()
        alldata.loc[~np.isfinite(alldata[c]),c] = np.nan
        alldata.loc[~alldata[c].isnull(),c] = ss.fit_transform(alldata.loc[~alldata[c].isnull(),c].values.reshape(-1,1))
        alldata[c].fillna(alldata[c].mean(),inplace=True)
        
    train = alldata[:ntrainrows]
    test = alldata[ntrainrows:]
    
    traintargets = train.TARGET.values
    train = GP2(train)
    test = GP2(test)
    train['TARGET'] = traintargets
    
    del train["TARGET"]
    train.to_csv("processed/train_gp2_features.csv", index = False)
    test.to_csv("processed/test_gp2_features.csv", index = False)