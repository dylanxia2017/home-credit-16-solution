# This script was taken from the public discussion https://www.kaggle.com/c/home-credit-default-risk/discussion/62983

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
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

def GP1(data):
    v = pd.DataFrame()
    v["i0"] = 0.059995*np.tanh(((((((((np.maximum(((((data["REFUSED_CNT_PAYMENT_SUM"]) + (1.570796)))), ((-1.0)))) + (((data["DAYS_BIRTH"]) * 2.0)))) + (((data["REGION_RATING_CLIENT"]) + (data["NAME_INCOME_TYPE_Working"]))))) * 2.0)) * 2.0)) 
    v["i1"] = 0.019540*np.tanh((((13.69431781768798828)) * (((data["REGION_RATING_CLIENT_W_CITY"]) + (np.tanh((np.maximum(((data["REGION_RATING_CLIENT_W_CITY"])), (((((1.0) + (((np.minimum(((data["REFUSED_DAYS_DECISION_MAX"])), ((data["REFUSED_CNT_PAYMENT_SUM"])))) + (data["REFUSED_DAYS_DECISION_MAX"]))))/2.0))))))))))) 
    v["i2"] = 0.099540*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((((((data["REGION_RATING_CLIENT_W_CITY"]) - (np.tanh((data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]))))) + (((((((data["DAYS_BIRTH"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) * 2.0)) - (data["CODE_GENDER"]))))) * 2.0)))) 
    v["i3"] = 0.097305*np.tanh(((((((np.maximum(((data["REFUSED_CNT_PAYMENT_SUM"])), ((((np.where(data["REFUSED_DAYS_DECISION_MEAN"]>0, np.where(data["REFUSED_DAYS_DECISION_MAX"]>0, 2.0, data["DAYS_BIRTH"] ), data["DAYS_BIRTH"] )) - (0.636620)))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i4"] = 0.099670*np.tanh(((((((((((np.tanh((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0)) - (data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) * 2.0)))) - (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) - (1.570796))) 
    v["i5"] = 0.099503*np.tanh(((((((np.where(data["REFUSED_DAYS_DECISION_MAX"]<0, data["REGION_RATING_CLIENT_W_CITY"], 2.0 )) + (((((data["FLAG_DOCUMENT_3"]) + (((data["DAYS_ID_PUBLISH"]) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))))) - (data["CODE_GENDER"]))))) * 2.0)) * 2.0)) 
    v["i6"] = 0.099849*np.tanh(((data["NEW_SCORES_STD"]) + (((((((data["REGION_RATING_CLIENT_W_CITY"]) + (((data["NAME_INCOME_TYPE_Working"]) + (((data["NEW_SCORES_STD"]) + (((data["REG_CITY_NOT_WORK_CITY"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))))))))) - (data["CODE_GENDER"]))) * 2.0)))) 
    v["i7"] = 0.099890*np.tanh(np.where((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (0.318310))/2.0)>0, (5.78559446334838867), np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] < -99998, data["NEW_DOC_IND_KURT"], ((((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))) * 2.0)) * 2.0)) * 2.0) ) )) 
    v["i8"] = 0.099495*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, (5.0), np.where(data["NEW_SCORES_STD"]>0, data["NEW_SCORES_STD"], np.where(data["NEW_SCORES_STD"]<0, ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))) * ((8.84948539733886719))), (13.10789012908935547) ) ) )) 
    v["i9"] = 0.099966*np.tanh(((np.where(((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * 2.0)) + (0.318310))>0, 3.0, ((data["REFUSED_DAYS_DECISION_MAX"]) + (((((data["REFUSED_DAYS_DECISION_MEAN"]) + (data["REFUSED_CNT_PAYMENT_SUM"]))) - (data["REFUSED_HOUR_APPR_PROCESS_START_MEAN"])))) )) * 2.0)) 
    v["i10"] = 0.099804*np.tanh((((((((data["DAYS_ID_PUBLISH"]) + (data["REGION_RATING_CLIENT_W_CITY"]))) + (((((data["REG_CITY_NOT_LIVE_CITY"]) - (np.where(data["NEW_CAR_TO_BIRTH_RATIO"]>0, 3.141593, np.tanh((data["CLOSED_DAYS_CREDIT_VAR"])) )))) * 2.0)))/2.0)) * 2.0)) 
    v["i11"] = 0.099970*np.tanh(((np.where(np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))>0, (6.61570215225219727), ((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * 2.0), data["DAYS_BIRTH"] )) - (data["CODE_GENDER"])) )) * 2.0)) 
    v["i12"] = 0.099924*np.tanh((-1.0*((np.where(np.minimum(((data["NAME_FAMILY_STATUS_Married"])), ((data["NEW_CAR_TO_EMPLOY_RATIO"])))>0, (10.0), (((6.0)) * (np.where(data["NEW_DOC_IND_KURT"]<0, 3.0, ((data["NEW_DOC_IND_KURT"]) + (data["EXT_SOURCE_1"])) ))) ))))) 
    v["i13"] = 0.099965*np.tanh(((((3.0) * (((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (np.minimum(((np.maximum(((data["FLOORSMAX_AVG"])), ((data["ENTRANCES_MEDI"]))))), ((((((((data["FLOORSMAX_AVG"]) / 2.0)) / 2.0)) - (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))))))))) * 2.0)) 
    v["i14"] = 0.099979*np.tanh(np.where(((data["ACTIVE_DAYS_CREDIT_VAR"]) - (data["CLOSED_CNT_CREDIT_PROLONG_SUM"]))<0, ((((data["ACTIVE_DAYS_CREDIT_VAR"]) - (data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) + (3.0)), (((((-1.0*((data["CLOSED_AMT_CREDIT_SUM_MEAN"])))) * 2.0)) - (data["ACTIVE_DAYS_CREDIT_VAR"])) )) 
    v["i15"] = 0.099860*np.tanh(((np.where(np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) + ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))*1.))))))>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], (-1.0*((data["NAME_FAMILY_STATUS_Married"]))) )) - (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0)))) 
    v["i16"] = 0.097603*np.tanh(((((data["DAYS_REGISTRATION"]) + (((data["REGION_RATING_CLIENT"]) - (data["NAME_INCOME_TYPE_State_servant"]))))) - (np.maximum(((((data["CODE_GENDER"]) - (data["OCCUPATION_TYPE_Laborers"])))), ((((((data["OCCUPATION_TYPE_Core_staff"]) * 2.0)) - (data["REGION_RATING_CLIENT_W_CITY"])))))))) 
    v["i17"] = 0.100000*np.tanh(np.maximum(((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) * 2.0)) * 2.0))), (((((data["ACTIVE_DAYS_CREDIT_VAR"]) + (((((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) * ((((np.tanh((data["ACTIVE_DAYS_CREDIT_VAR"]))) > (data["FLAG_DOCUMENT_3"]))*1.)))) - (data["NEW_CAR_TO_BIRTH_RATIO"]))))/2.0))))) 
    v["i18"] = 0.099955*np.tanh(np.where(((data["EXT_SOURCE_1"]) + (((data["CLOSED_DAYS_CREDIT_VAR"]) * (data["NAME_FAMILY_STATUS_Married"])))) < -99998, data["ORGANIZATION_TYPE_Self_employed"], ((data["ORGANIZATION_TYPE_Self_employed"]) - ((((5.0)) * (((1.0) + (data["EXT_SOURCE_1"])))))) )) 
    v["i19"] = 0.099900*np.tanh(((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["OCCUPATION_TYPE_Drivers"] )) + ((((((((data["ORGANIZATION_TYPE_Construction"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"]))/2.0)) + (data["OCCUPATION_TYPE_Drivers"]))) + ((((data["FLAG_PHONE"]) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)))))) 
    v["i20"] = 0.099957*np.tanh(((np.tanh((data["REFUSED_CNT_PAYMENT_SUM"]))) + (np.maximum((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (data["FLAG_WORK_PHONE"]))/2.0))), ((((np.maximum(((data["NEW_SCORES_STD"])), ((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((data["DAYS_ID_PUBLISH"]))))))) - (0.636620)))))))) 
    v["i21"] = 0.099700*np.tanh(np.where(data["CC_AMT_DRAWINGS_CURRENT_VAR"] < -99998, 0.318310, (((((((((-1.0*((data["CC_NAME_CONTRACT_STATUS_Active_VAR"])))) + (data["CC_AMT_DRAWINGS_CURRENT_VAR"]))) + (data["CC_AMT_DRAWINGS_CURRENT_VAR"]))) - (data["CC_NAME_CONTRACT_STATUS_Demand_VAR"]))) * ((((11.63801860809326172)) * 2.0))) )) 
    v["i22"] = 0.099600*np.tanh(((((np.where(data["ORGANIZATION_TYPE_Military"]>0, -2.0, ((data["REG_CITY_NOT_LIVE_CITY"]) - (((((np.maximum(((data["NAME_INCOME_TYPE_State_servant"])), ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))) * 2.0)) * 2.0))) )) + (data["NAME_EDUCATION_TYPE_Lower_secondary"]))) + (data["NAME_EDUCATION_TYPE_Lower_secondary"]))) 
    v["i23"] = 0.099700*np.tanh(((((-1.0*((data["OCCUPATION_TYPE_Accountants"])))) + ((((data["NAME_HOUSING_TYPE_Municipal_apartment"]) + (np.minimum(((((((((data["NAME_FAMILY_STATUS_Married"]) - (data["OCCUPATION_TYPE_Sales_staff"]))) * (data["ACTIVE_DAYS_CREDIT_VAR"]))) / 2.0))), (((-1.0*((data["ORGANIZATION_TYPE_Industry__type_9"]))))))))/2.0)))/2.0)) 
    v["i24"] = 0.097890*np.tanh(np.where(((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (data["NAME_INCOME_TYPE_Unemployed"]))<0, ((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((((data["ORGANIZATION_TYPE_Transport__type_3"]) * 2.0))))) - (np.tanh((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (-1.0)))))), (6.0) )) 
    v["i25"] = 0.100000*np.tanh(((data["ORGANIZATION_TYPE_Transport__type_3"]) + (((((((((((((((((data["INSTAL_DPD_MEAN"]) - (data["ORGANIZATION_TYPE_Military"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) 
    v["i26"] = 0.099698*np.tanh((((np.where((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) < (data["ACTIVE_AMT_ANNUITY_MAX"]))*1.)>0, (8.0), ((((data["OCCUPATION_TYPE_Laborers"]) - (data["WALLSMATERIAL_MODE_Panel"]))) - (data["FLAG_PHONE"])) )) + (np.maximum(((data["OCCUPATION_TYPE_Low_skill_Laborers"])), ((data["OCCUPATION_TYPE_Cooking_staff"])))))/2.0)) 
    v["i27"] = 0.099970*np.tanh(((data["ORGANIZATION_TYPE_Trade__type_3"]) - (np.maximum((((((((((((((data["ORGANIZATION_TYPE_Trade__type_3"]) + (data["CC_NAME_CONTRACT_STATUS_Completed_VAR"]))/2.0)) > (np.tanh((np.tanh((data["CC_AMT_DRAWINGS_CURRENT_VAR"]))))))*1.)) + (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]))/2.0)) * 2.0))), ((data["OCCUPATION_TYPE_High_skill_tech_staff"])))))) 
    v["i28"] = 0.099996*np.tanh(((((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((((-1.0) / 2.0))))) + (((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) - (np.where(data["ORGANIZATION_TYPE_Trade__type_2"]<0, ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0)) * 2.0), data["NEW_CREDIT_TO_INCOME_RATIO"] )))))) * 2.0)) 
    v["i29"] = 0.100000*np.tanh((((13.18957233428955078)) * (((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) + (((((np.maximum(((3.0)), (((((14.53842449188232422)) * (data["INSTAL_DPD_MEAN"])))))) * (data["INSTAL_DPD_MEAN"]))) * 2.0)))) * ((5.49084424972534180)))))) 
    v["i30"] = 0.097617*np.tanh(((np.maximum((((((((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN"]) * 2.0)) > (np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])))))*1.))), ((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]))) + (((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) * 2.0))))))) * 2.0)) 
    v["i31"] = 0.099942*np.tanh((((5.04678153991699219)) * (np.where(((data["EXT_SOURCE_3"]) + (data["ORGANIZATION_TYPE_Realtor"])) < -99998, data["OCCUPATION_TYPE_Cleaning_staff"], (((-1.0*((np.tanh((np.tanh((0.318310)))))))) - (data["EXT_SOURCE_3"])) )))) 
    v["i32"] = 0.099999*np.tanh(((((((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), (((-1.0*((data["HOUR_APPR_PROCESS_START"]))))))) - (((((data["NAME_EDUCATION_TYPE_Incomplete_higher"]) * 2.0)) * 2.0)))) - (data["ORGANIZATION_TYPE_Security_Ministries"]))) - (np.tanh((data["OCCUPATION_TYPE_Core_staff"]))))) 
    v["i33"] = 0.099756*np.tanh(np.where(data["YEARS_BEGINEXPLUATATION_MODE"] < -99998, ((np.minimum(((data["AMT_INCOME_TOTAL"])), (((((-1.0*((data["ORGANIZATION_TYPE_Industry__type_9"])))) * (data["ORGANIZATION_TYPE_Industry__type_9"])))))) * 2.0), (((data["NAME_INCOME_TYPE_Unemployed"]) > (data["YEARS_BEGINEXPLUATATION_MODE"]))*1.) )) 
    v["i34"] = 0.099040*np.tanh(((((((data["ORGANIZATION_TYPE_Realtor"]) + (data["FLAG_WORK_PHONE"]))/2.0)) + (((np.maximum(((data["NAME_INCOME_TYPE_Unemployed"])), ((data["OCCUPATION_TYPE_Security_staff"])))) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) - ((((((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) / 2.0)) > (data["ORGANIZATION_TYPE_Realtor"]))*1.)))))))/2.0)) 
    v["i35"] = 0.099800*np.tanh((((((data["ORGANIZATION_TYPE_Realtor"]) + (((((((data["ORGANIZATION_TYPE_Restaurant"]) + (data["NAME_INCOME_TYPE_Maternity_leave"]))/2.0)) + (data["NAME_INCOME_TYPE_Unemployed"]))/2.0)))/2.0)) - (np.where(data["ORGANIZATION_TYPE_Restaurant"]>0, data["WEEKDAY_APPR_PROCESS_START_SUNDAY"], (((data["ORGANIZATION_TYPE_Industry__type_9"]) + (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]))/2.0) )))) 
    v["i36"] = 0.099500*np.tanh((((((data["ORGANIZATION_TYPE_Self_employed"]) / 2.0)) + ((((-1.0*((np.maximum(((np.where(data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]>0, data["OCCUPATION_TYPE_Accountants"], data["OCCUPATION_TYPE_Medicine_staff"] ))), (((((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]) + (data["ORGANIZATION_TYPE_Military"]))/2.0)))))))) - (data["OCCUPATION_TYPE_Accountants"]))))/2.0)) 
    v["i37"] = 0.099450*np.tanh(np.where((((data["CLOSED_DAYS_CREDIT_MEAN"]) > (1.570796))*1.)>0, 3.0, np.where(data["AMT_INCOME_TOTAL"]>0, data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"], np.where(data["WEEKDAY_APPR_PROCESS_START_MONDAY"]>0, -1.0, (((data["CLOSED_DAYS_CREDIT_MEAN"]) > (1.570796))*1.) ) ) )) 
    v["i38"] = 0.098500*np.tanh(np.where(((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (np.tanh((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]))))>0, (-1.0*(((10.21496009826660156)))), np.maximum(((0.318310)), ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + ((10.42618179321289062)))))) )) 
    v["i39"] = 0.097940*np.tanh((-1.0*((((data["ORGANIZATION_TYPE_Police"]) - (((data["ORGANIZATION_TYPE_Realtor"]) + (((data["ORGANIZATION_TYPE_Construction"]) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((data["ORGANIZATION_TYPE_Construction"]) + (data["NAME_INCOME_TYPE_Unemployed"]))))/2.0))))))))))) 
    v["i40"] = 0.099370*np.tanh(((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]>0, ((((((0.318310) - (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))) * 2.0)) * 2.0), (((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_EDUCATION_TYPE_Lower_secondary"]))/2.0)) - (data["ORGANIZATION_TYPE_Bank"])) )) * 2.0)) 
    v["i41"] = 0.099963*np.tanh(((((((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * ((-1.0*((data["AMT_INCOME_TOTAL"])))))) * 2.0)) * 2.0)) - (np.maximum(((data["AMT_INCOME_TOTAL"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"])))))) - (data["NAME_HOUSING_TYPE_Office_apartment"]))) * 2.0)) * 2.0)) 
    v["i42"] = 0.099999*np.tanh((((((-1.0*((((data["BURO_STATUS_0_MEAN_MEAN"]) + (np.where(data["BURO_STATUS_0_MEAN_MEAN"]>0, data["BURO_MONTHS_BALANCE_SIZE_MEAN"], (((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) > (-1.0))*1.) ))))))) * ((((data["BURO_STATUS_0_MEAN_MEAN"]) > (-1.0))*1.)))) * 2.0)) 
    v["i43"] = 0.093012*np.tanh((((((-1.0*((((((((((-1.0*((data["ORGANIZATION_TYPE_Trade__type_2"])))) - (np.maximum(((data["NEW_CAR_TO_EMPLOY_RATIO"])), ((data["ORGANIZATION_TYPE_Industry__type_12"])))))) < (data["REG_REGION_NOT_LIVE_REGION"]))*1.)) * 2.0))))) - (data["REG_REGION_NOT_LIVE_REGION"]))) * 2.0)) 
    v["i44"] = 0.082010*np.tanh(np.where(data["AMT_INCOME_TOTAL"]<0, ((((((((1.570796) * (np.where(data["NAME_HOUSING_TYPE_Rented_apartment"]>0, data["NAME_HOUSING_TYPE_Rented_apartment"], data["AMT_INCOME_TOTAL"] )))) - (data["NAME_FAMILY_STATUS_Widow"]))) * 2.0)) - (data["NAME_FAMILY_STATUS_Widow"])), data["NAME_HOUSING_TYPE_Rented_apartment"] )) 
    v["i45"] = 0.098100*np.tanh(((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (data["NAME_INCOME_TYPE_Unemployed"]))/2.0)) + (((((data["OCCUPATION_TYPE_Cleaning_staff"]) - (data["ORGANIZATION_TYPE_Industry__type_5"]))) - ((((data["ORGANIZATION_TYPE_School"]) + (data["NAME_EDUCATION_TYPE_Academic_degree"]))/2.0)))))/2.0)) 
    v["i46"] = 0.099505*np.tanh((((((-1.0*(((((data["NAME_INCOME_TYPE_Maternity_leave"]) > ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + ((((((data["AMT_INCOME_TOTAL"]) / 2.0)) + (2.0))/2.0)))/2.0)))*1.))))) - (data["ORGANIZATION_TYPE_Hotel"]))) - (((data["NAME_EDUCATION_TYPE_Academic_degree"]) / 2.0)))) 
    v["i47"] = 0.099995*np.tanh(((((data["FLAG_OWN_REALTY"]) * (data["ORGANIZATION_TYPE_Trade__type_7"]))) + (np.maximum((((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_INCOME_TYPE_Maternity_leave"]))/2.0))), ((np.maximum(((((data["ORGANIZATION_TYPE_Trade__type_3"]) + (data["ORGANIZATION_TYPE_Legal_Services"])))), ((data["ORGANIZATION_TYPE_Trade__type_7"]))))))))) 
    v["i48"] = 0.098997*np.tanh((((data["NAME_TYPE_SUITE_Children"]) + (((((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_FAMILY_STATUS_Separated"]))/2.0)) + (np.where(data["NAME_FAMILY_STATUS_Separated"]<0, np.maximum(((data["NAME_HOUSING_TYPE_Municipal_apartment"])), ((data["NAME_TYPE_SUITE_Other_B"]))), ((data["AMT_INCOME_TOTAL"]) - (data["NAME_TYPE_SUITE_Other_B"])) )))/2.0)))/2.0)) 
    v["i49"] = 0.071280*np.tanh((((((((-1.0*((data["CODE_GENDER"])))) + (data["ORGANIZATION_TYPE_Transport__type_4"]))/2.0)) + (((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + ((((data["CODE_GENDER"]) + (data["ORGANIZATION_TYPE_Realtor"]))/2.0)))/2.0)) + (((data["OCCUPATION_TYPE_Low_skill_Laborers"]) - (data["OCCUPATION_TYPE_Private_service_staff"]))))/2.0)))/2.0)) 
    v["i50"] = 0.090101*np.tanh((((data["OCCUPATION_TYPE_Waiters_barmen_staff"]) + (((data["NAME_INCOME_TYPE_Maternity_leave"]) - (((((((0.318310) < (np.where(data["NEW_INC_PER_CHLD"]>0, data["AMT_INCOME_TOTAL"], data["ORGANIZATION_TYPE_Security_Ministries"] )))*1.)) + ((((data["NEW_INC_BY_ORG"]) + (data["ORGANIZATION_TYPE_Security_Ministries"]))/2.0)))/2.0)))))/2.0)) 
    v["i51"] = 0.077600*np.tanh(np.where((((data["AMT_INCOME_TOTAL"]) + (data["HOUSETYPE_MODE_specific_housing"]))/2.0)>0, np.maximum(((np.minimum(((data["ORGANIZATION_TYPE_Business_Entity_Type_3"])), ((data["HOUSETYPE_MODE_specific_housing"]))))), ((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (np.maximum(((data["HOUSETYPE_MODE_specific_housing"])), ((data["AMT_INCOME_TOTAL"])))))))), data["ORGANIZATION_TYPE_Legal_Services"] )) 
    v["i52"] = 0.084003*np.tanh((((((((((data["ORGANIZATION_TYPE_Legal_Services"]) > (((data["NEW_INC_PER_CHLD"]) + (np.tanh((1.0))))))*1.)) + (data["ORGANIZATION_TYPE_Realtor"]))) - (data["ORGANIZATION_TYPE_Industry__type_12"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) 
    v["i53"] = 0.096301*np.tanh((((((((((((data["AMT_INCOME_TOTAL"]) + (data["REFUSED_AMT_DOWN_PAYMENT_MAX"]))) + (data["REFUSED_AMT_DOWN_PAYMENT_MAX"]))) * (np.tanh((0.318310))))) > (np.tanh((data["REFUSED_AMT_DOWN_PAYMENT_MEAN"]))))*1.)) + (data["HOUSETYPE_MODE_specific_housing"]))) 
    v["i54"] = 0.098989*np.tanh(np.minimum((((((3.0)) - (data["REFUSED_AMT_ANNUITY_MAX"])))), ((np.minimum((((((0.0) < (data["REFUSED_APP_CREDIT_PERC_MAX"]))*1.))), ((((data["REFUSED_AMT_ANNUITY_MAX"]) - (((data["REFUSED_APP_CREDIT_PERC_MAX"]) + (data["REFUSED_AMT_GOODS_PRICE_MEAN"]))))))))))) 
    v["i55"] = 0.094598*np.tanh(np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((np.where((((((data["AMT_INCOME_TOTAL"]) < (data["ORGANIZATION_TYPE_Trade__type_6"]))*1.)) * (data["NAME_INCOME_TYPE_Maternity_leave"]))<0, data["NAME_INCOME_TYPE_State_servant"], (-1.0*((np.maximum(((data["NAME_INCOME_TYPE_State_servant"])), ((data["ORGANIZATION_TYPE_Transport__type_3"])))))) ))))) 
    v["i56"] = 0.089650*np.tanh(((((np.minimum(((((data["OCCUPATION_TYPE_Managers"]) * (((np.tanh((((data["NAME_TYPE_SUITE_Spouse__partner"]) + (np.tanh((data["AMT_INCOME_TOTAL"]))))))) + (data["NAME_EDUCATION_TYPE_Academic_degree"])))))), (((-1.0*((data["ORGANIZATION_TYPE_Trade__type_6"]))))))) * 2.0)) * 2.0)) 
    v["i57"] = 0.100000*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] < -99998, data["ORGANIZATION_TYPE_Legal_Services"], np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]>0, ((((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]))) - (data["AMT_INCOME_TOTAL"]))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])), data["NEW_INC_PER_CHLD"] ) )) 
    v["i58"] = 0.099530*np.tanh((((((((((((data["AMT_INCOME_TOTAL"]) + (data["NAME_INCOME_TYPE_Unemployed"]))/2.0)) + (((data["ORGANIZATION_TYPE_Restaurant"]) - (data["ORGANIZATION_TYPE_Electricity"]))))/2.0)) - (data["ORGANIZATION_TYPE_Industry__type_5"]))) + ((((data["AMT_INCOME_TOTAL"]) > (((2.0) / 2.0)))*1.)))/2.0)) 
    v["i59"] = 0.099529*np.tanh(((((((((data["OCCUPATION_TYPE_IT_staff"]) * (data["ORGANIZATION_TYPE_Industry__type_12"]))) - ((((((np.minimum(((data["OCCUPATION_TYPE_IT_staff"])), ((data["NEW_INC_PER_CHLD"])))) / 2.0)) > (((data["FONDKAPREMONT_MODE_org_spec_account"]) * (data["NAME_EDUCATION_TYPE_Academic_degree"]))))*1.)))) * 2.0)) * 2.0)) 
    v["i60"] = 0.099740*np.tanh(((((data["DEF_60_CNT_SOCIAL_CIRCLE"]) + (((np.minimum(((data["ORGANIZATION_TYPE_Police"])), ((np.minimum(((data["FLAG_EMP_PHONE"])), ((data["NEW_INC_PER_CHLD"]))))))) * (np.where(data["ORGANIZATION_TYPE_Transport__type_1"]>0, data["ORGANIZATION_TYPE_Transport__type_1"], data["ORGANIZATION_TYPE_Police"] )))))) + (data["NAME_INCOME_TYPE_Unemployed"]))) 
    v["i61"] = 0.099555*np.tanh((-1.0*((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]<0, np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, (-1.0*((np.maximum(((data["DAYS_REGISTRATION"])), ((0.0)))))), data["ORGANIZATION_TYPE_Police"] ), ((data["DAYS_REGISTRATION"]) + (((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) / 2.0))) ))))) 
    v["i62"] = 0.100000*np.tanh(((((((-1.0) / 2.0)) - (np.where(data["EXT_SOURCE_3"] < -99998, 0.0, ((data["EXT_SOURCE_3"]) * 2.0) )))) - ((((np.tanh((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) / 2.0)))) < (data["EXT_SOURCE_3"]))*1.)))) 
    v["i63"] = 0.098850*np.tanh(np.where(((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > ((-1.0*((0.318310)))))*1.)) < (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) - (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))))*1.)>0, -1.0, ((((-1.0*((0.318310)))) > (data["REFUSED_AMT_ANNUITY_MEAN"]))*1.) )) 
    v["i64"] = 0.099952*np.tanh((-1.0*((((np.where(data["EXT_SOURCE_3"] < -99998, np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]<0, 0.318310, data["EXT_SOURCE_3"] ), ((data["EXT_SOURCE_3"]) - (np.tanh((((np.tanh((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))) * 2.0))))) )) * 2.0))))) 
    v["i65"] = 0.099900*np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, 0.318310, (-1.0*((((((data["NEW_SOURCES_PROD"]) - ((-1.0*(((((((data["NEW_SOURCES_PROD"]) > (-1.0))*1.)) - (data["ORGANIZATION_TYPE_XNA"])))))))) * ((8.0)))))) )) 
    v["i66"] = 0.095998*np.tanh(np.minimum(((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.636620, ((data["ACTIVE_DAYS_CREDIT_VAR"]) / 2.0) ))), ((np.where(((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["ACTIVE_DAYS_CREDIT_VAR"])) < -99998, (-1.0*((data["REFUSED_DAYS_DECISION_MEAN"]))), data["NEW_CREDIT_TO_INCOME_RATIO"] ))))) 
    v["i67"] = 0.099733*np.tanh((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.)) - (np.where(data["EXT_SOURCE_3"] < -99998, ((((1.0)) > (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.), data["EXT_SOURCE_3"] )))) - ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)))) 
    v["i68"] = 0.098499*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.318310, ((((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"])))) * 2.0)) - (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))))) * 2.0) )) 
    v["i69"] = 0.099999*np.tanh(np.where((((((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0)) + ((((3.0) > (data["NEW_SCORES_STD"]))*1.)))/2.0)<0, np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]>0, (8.0), data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"] ), ((((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0)) * 2.0) )) 
    v["i70"] = 0.099998*np.tanh(((((data["NEW_SCORES_STD"]) / 2.0)) - (np.minimum((((((np.tanh((data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]))) + (3.141593))/2.0))), ((np.maximum(((((((((data["NEW_SCORES_STD"]) * 2.0)) * 2.0)) * 2.0))), ((data["NEW_SCORES_STD"]))))))))) 
    v["i71"] = 0.099949*np.tanh(np.where((((data["NEW_SOURCES_PROD"]) + (np.tanh((1.0))))/2.0)<0, np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (12.80127429962158203) ), (-1.0*(((12.80127429962158203)))) )) 
    v["i72"] = 0.099500*np.tanh((((((data["NEW_SOURCES_PROD"]) > (((-2.0) * 2.0)))*1.)) * ((((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)) - (0.636620))) - (data["NEW_SOURCES_PROD"]))) * 2.0)))) 
    v["i73"] = 0.099052*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, (((0.318310) > (((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) + (((((((-1.0*((data["REFUSED_CNT_PAYMENT_MEAN"])))) > (0.318310))*1.)) / 2.0)))))*1.), (((data["ACTIVE_DAYS_CREDIT_VAR"]) + (data["ORGANIZATION_TYPE_XNA"]))/2.0) )) 
    v["i74"] = 0.099260*np.tanh((((((-1.0*(((((data["EXT_SOURCE_1"]) > (((np.minimum((((((data["NEW_SOURCES_PROD"]) < (((0.636620) * 2.0)))*1.))), ((0.636620)))) + (np.tanh((0.318310))))))*1.))))) * 2.0)) * 2.0)) 
    v["i75"] = 0.099504*np.tanh((-1.0*((np.where(np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]<0, data["EXT_SOURCE_1"], 0.318310 ) < -99998, 0.318310, np.where(data["NEW_SOURCES_PROD"] < -99998, data["EXT_SOURCE_1"], np.where(data["NEW_SOURCES_PROD"]<0, data["NEW_SOURCES_PROD"], data["EXT_SOURCE_1"] ) ) ))))) 
    v["i76"] = 0.099849*np.tanh(np.where((((0.636620) < (((data["DAYS_BIRTH"]) / 2.0)))*1.)>0, -2.0, (((((data["DAYS_BIRTH"]) * (data["DAYS_BIRTH"]))) < (np.where(data["DAYS_BIRTH"]>0, 0.636620, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] )))*1.) )) 
    v["i77"] = 0.099975*np.tanh((((-1.0*((np.where(data["REFUSED_CNT_PAYMENT_MEAN"]>0, data["REFUSED_AMT_CREDIT_MEAN"], (((((np.minimum(((np.maximum(((np.tanh((data["REFUSED_AMT_ANNUITY_MAX"])))), ((data["REFUSED_AMT_ANNUITY_MAX"]))))), ((data["REFUSED_AMT_APPLICATION_MAX"])))) < (data["REFUSED_AMT_ANNUITY_MEAN"]))*1.)) * 2.0) ))))) * 2.0)) 
    v["i78"] = 0.099301*np.tanh(np.minimum(((((1.0) - (data["DAYS_BIRTH"])))), ((np.where(data["OCCUPATION_TYPE_Laborers"]>0, np.maximum(((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])), ((data["DAYS_BIRTH"]))), (((((data["NEW_SCORES_STD"]) / 2.0)) < (data["NEW_INC_PER_CHLD"]))*1.) ))))) 
    v["i79"] = 0.097599*np.tanh(np.minimum(((np.where(data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]<0, ((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) - (data["NAME_FAMILY_STATUS_Married"]))) / 2.0)) / 2.0), data["NAME_FAMILY_STATUS_Married"] ))), ((((data["NAME_FAMILY_STATUS_Married"]) - (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])))))) 
    v["i80"] = 0.094203*np.tanh(np.where(data["AMT_INCOME_TOTAL"]<0, ((data["NEW_DOC_IND_KURT"]) * (data["AMT_INCOME_TOTAL"])), ((data["NEW_DOC_IND_KURT"]) + ((((np.minimum(((data["FONDKAPREMONT_MODE_reg_oper_spec_account"])), ((data["OCCUPATION_TYPE_Core_staff"])))) + (((data["OCCUPATION_TYPE_Core_staff"]) - (data["FONDKAPREMONT_MODE_reg_oper_spec_account"]))))/2.0))) )) 
    v["i81"] = 0.099984*np.tanh(np.where(np.minimum(((data["DAYS_ID_PUBLISH"])), ((np.minimum(((data["NEW_DOC_IND_KURT"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"]))))))<0, np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["DAYS_ID_PUBLISH"], ((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["NEW_DOC_IND_KURT"])) ), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] )) 
    v["i82"] = 0.099600*np.tanh((((2.0) < ((((((((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (data["FLAG_CONT_MOBILE"]))) * 2.0)) + (3.0))) + (((((0.33989319205284119)) + (np.maximum(((data["DAYS_ID_PUBLISH"])), ((data["AMT_INCOME_TOTAL"])))))/2.0)))/2.0)))*1.)) 
    v["i83"] = 0.073300*np.tanh(((((((data["WALLSMATERIAL_MODE_Others"]) + ((((data["ORGANIZATION_TYPE_Realtor"]) + (0.636620))/2.0)))/2.0)) + ((((-1.0*((data["NAME_EDUCATION_TYPE_Academic_degree"])))) + (((data["ORGANIZATION_TYPE_Mobile"]) - ((((0.318310) < (data["AMT_INCOME_TOTAL"]))*1.)))))))/2.0)) 
    v["i84"] = 0.081498*np.tanh(((((((data["NAME_EDUCATION_TYPE_Higher_education"]) + (((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) * (data["AMT_INCOME_TOTAL"]))))/2.0)) + (((data["AMT_INCOME_TOTAL"]) * (((data["AMT_INCOME_TOTAL"]) * (((data["AMT_INCOME_TOTAL"]) * ((-1.0*((data["NAME_EDUCATION_TYPE_Higher_education"])))))))))))/2.0)) 
    v["i85"] = 0.099970*np.tanh(((data["ORGANIZATION_TYPE_Legal_Services"]) + (np.where((((2.0) < (((data["ORGANIZATION_TYPE_Self_employed"]) + (data["NEW_INC_PER_CHLD"]))))*1.)>0, (((data["NEW_INC_PER_CHLD"]) < (((data["ORGANIZATION_TYPE_Legal_Services"]) * 2.0)))*1.), data["NAME_INCOME_TYPE_Maternity_leave"] )))) 
    v["i86"] = 0.098020*np.tanh(np.where(data["REGION_RATING_CLIENT_W_CITY"]>0, np.minimum(((((data["ORGANIZATION_TYPE_Agriculture"]) - (data["NEW_INC_PER_CHLD"])))), (((((data["ORGANIZATION_TYPE_Agriculture"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"]))/2.0)))), (((((data["NEW_INC_PER_CHLD"]) + (data["REGION_RATING_CLIENT"]))) > (1.570796))*1.) )) 
    v["i87"] = 0.094996*np.tanh(((((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.where(data["AMT_INCOME_TOTAL"]<0, np.where(data["NEW_INC_PER_CHLD"]<0, ((data["ORGANIZATION_TYPE_Industry__type_2"]) / 2.0), data["REG_REGION_NOT_LIVE_REGION"] ), np.maximum(((data["ORGANIZATION_TYPE_Industry__type_2"])), ((data["REG_REGION_NOT_LIVE_REGION"]))) )))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) 
    v["i88"] = 0.099811*np.tanh(np.where(((((data["WEEKDAY_APPR_PROCESS_START_TUESDAY"]) * (data["WEEKDAY_APPR_PROCESS_START_TUESDAY"]))) - (data["AMT_CREDIT"]))>0, np.maximum(((data["AMT_INCOME_TOTAL"])), ((0.0))), ((data["WEEKDAY_APPR_PROCESS_START_TUESDAY"]) - (data["AMT_INCOME_TOTAL"])) )) 
    v["i89"] = 0.099800*np.tanh(((((((((((data["ORGANIZATION_TYPE_Transport__type_4"]) + ((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (data["ORGANIZATION_TYPE_Transport__type_4"]))) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)))/2.0)) > ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (0.318310))*1.)))*1.)) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) * 2.0)) 
    v["i90"] = 0.098490*np.tanh(np.maximum(((data["ORGANIZATION_TYPE_Industry__type_1"])), ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (np.where((((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, data["AMT_INCOME_TOTAL"], data["NEW_INC_PER_CHLD"] )) + (data["NAME_INCOME_TYPE_Unemployed"]))/2.0)>0, data["NEW_INC_PER_CHLD"], (6.0) ))))))) 
    v["i91"] = 0.093999*np.tanh(((data["NAME_INCOME_TYPE_Unemployed"]) + (((((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"]))/2.0)) - (data["NAME_INCOME_TYPE_Student"]))))/2.0)) / 2.0)) + (((data["NAME_INCOME_TYPE_Unemployed"]) - (data["ORGANIZATION_TYPE_Transport__type_1"]))))/2.0)))) 
    v["i92"] = 0.098000*np.tanh(np.minimum(((np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_State_servant"], np.where((((1.0) + (data["EXT_SOURCE_3"]))/2.0)>0, data["EXT_SOURCE_3"], (8.0) ) ))), ((((0.636620) - (data["EXT_SOURCE_3"])))))) 
    v["i93"] = 0.099730*np.tanh((((-1.0*((((((np.where((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) > ((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))*1.)))*1.)>0, data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"], (((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]))*1.) )) * 2.0)) * 2.0))))) * 2.0)) 
    v["i94"] = 0.098545*np.tanh(np.minimum(((np.where((((1.0) < (data["AMT_REQ_CREDIT_BUREAU_QRT"]))*1.)>0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"], ((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * (-2.0)) ))), (((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (np.tanh((data["AMT_REQ_CREDIT_BUREAU_QRT"]))))*1.))))) 
    v["i95"] = 0.095403*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, 0.318310, ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * (np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]>0, data["ACTIVE_CNT_CREDIT_PROLONG_SUM"], (-1.0*((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))) )>0, 3.141593, (-1.0*((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))) ))) )) 
    v["i96"] = 0.098999*np.tanh(((((np.maximum(((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) - (data["CLOSED_AMT_CREDIT_SUM_SUM"]))) * 2.0))), ((np.minimum(((data["CLOSED_DAYS_CREDIT_MEAN"])), ((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))))))) * 2.0)) - ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) > (1.0))*1.)))) 
    v["i97"] = 0.076491*np.tanh(np.where(np.where(data["NEW_SCORES_STD"]>0, data["DAYS_BIRTH"], data["ENTRANCES_MODE"] )>0, (((((data["DAYS_BIRTH"]) < (data["ENTRANCES_MODE"]))*1.)) - (data["DAYS_BIRTH"])), ((np.tanh((np.tanh((data["DAYS_BIRTH"]))))) / 2.0) )) 
    v["i98"] = 0.049979*np.tanh(((np.where((((((data["CLOSED_AMT_ANNUITY_MEAN"]) + (data["NAME_INCOME_TYPE_Student"]))) + (data["ORGANIZATION_TYPE_Hotel"]))/2.0)<0, ((data["NAME_INCOME_TYPE_Working"]) * (data["ORGANIZATION_TYPE_Hotel"])), data["NAME_INCOME_TYPE_Working"] )) * 2.0)) 
    v["i99"] = 0.071179*np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, (-1.0*(((((((0.0) > (data["AMT_INCOME_TOTAL"]))*1.)) / 2.0)))), ((np.tanh((np.where(data["EXT_SOURCE_3"]>0, 0.636620, data["EXT_SOURCE_3"] )))) - (data["EXT_SOURCE_3"])) )) 
    v["i100"] = 0.095300*np.tanh(np.minimum((((((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) < (data["AMT_REQ_CREDIT_BUREAU_YEAR"]))*1.))), ((((np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), ((np.maximum((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) + (data["AMT_REQ_CREDIT_BUREAU_QRT"]))/2.0))), ((0.318310))))))) - (data["AMT_REQ_CREDIT_BUREAU_YEAR"])))))) 
    v["i101"] = 0.097997*np.tanh(np.tanh((np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]<0, np.where((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)) + (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))>0, (-1.0*((0.636620))), 0.318310 ), data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"] )))) 
    v["i102"] = 0.091041*np.tanh(np.minimum((((((-1.0*((np.maximum(((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["DAYS_BIRTH"]))))))) + ((((-1.0*((data["ORGANIZATION_TYPE_Hotel"])))) + (1.0)))))), (((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0)) > (data["ORGANIZATION_TYPE_Hotel"]))*1.))))) 
    v["i103"] = 0.098532*np.tanh(np.where(data["ORGANIZATION_TYPE_Medicine"]>0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"], (((-1.0*((((np.minimum(((0.318310)), ((np.where(data["NAME_INCOME_TYPE_Maternity_leave"]>0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NEW_INC_PER_CHLD"] ))))) * (data["NAME_FAMILY_STATUS_Single___not_married"])))))) / 2.0) )) 
    v["i104"] = 0.080961*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]<0, (0.16092185676097870), data["ACTIVE_DAYS_CREDIT_VAR"] ), np.where((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) + (0.318310))/2.0)>0, (((data["ORGANIZATION_TYPE_Mobile"]) + (0.318310))/2.0), data["ACTIVE_DAYS_CREDIT_VAR"] ) )) 
    v["i105"] = 0.095005*np.tanh(np.where((((data["EXT_SOURCE_3"]) + (-1.0))/2.0)<0, np.maximum(((((data["WALLSMATERIAL_MODE_Stone__brick"]) * (data["AMT_INCOME_TOTAL"])))), ((data["NAME_INCOME_TYPE_Maternity_leave"]))), ((((data["CLOSED_AMT_ANNUITY_MEAN"]) + (data["WALLSMATERIAL_MODE_Stone__brick"]))) * 2.0) )) 
    v["i106"] = 0.096790*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]<0, ((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]>0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["NAME_INCOME_TYPE_Maternity_leave"] )) * 2.0), np.where(data["EXT_SOURCE_3"]<0, ((np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"])))) * 2.0), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] ) )) 
    v["i107"] = 0.019000*np.tanh((((9.0)) * ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) > (np.where(((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) - ((-1.0*((data["ACTIVE_DAYS_CREDIT_VAR"])))))) - ((-1.0*((data["CLOSED_DAYS_CREDIT_MEAN"]))))) < -99998, -2.0, data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"] )))*1.)))) 
    v["i108"] = 0.098660*np.tanh((((((data["EXT_SOURCE_3"]) - (((1.570796) - (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))))) > (np.where(data["ACTIVE_DAYS_CREDIT_VAR"]<0, np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]<0, data["ACTIVE_DAYS_CREDIT_VAR"], 3.0 ), ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0) )))*1.)) 
    v["i109"] = 0.099920*np.tanh(((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((((3.141593) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))) * 2.0))), (((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 2.0) )) * 2.0)) * 2.0)) 
    v["i110"] = 0.099820*np.tanh(np.minimum(((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["NAME_CONTRACT_TYPE_Cash_loans"])))), ((np.where(data["DAYS_BIRTH"]>0, data["NEW_DOC_IND_KURT"], np.where(data["AMT_CREDIT"]>0, ((data["NAME_CONTRACT_TYPE_Cash_loans"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"])), np.tanh((-1.0)) ) ))))) 
    v["i111"] = 0.099709*np.tanh(np.where(data["BURO_MONTHS_BALANCE_MAX_MAX"]<0, (((data["BURO_MONTHS_BALANCE_MIN_MIN"]) > (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.), np.tanh(((((((-1.0) - (data["BURO_STATUS_0_MEAN_MEAN"]))) + ((((-1.0) > (data["BURO_STATUS_0_MEAN_MEAN"]))*1.)))/2.0))) )) 
    v["i112"] = 0.099001*np.tanh(np.where(data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]>0, data["AMT_INCOME_TOTAL"], ((((((-1.0*((-1.0)))) / 2.0)) < (((np.where(data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] < -99998, data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"], 0.318310 )) - (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))))*1.) )) 
    v["i113"] = 0.099000*np.tanh(((np.minimum(((np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"]<0, (-1.0*((np.where(data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]<0, data["NEW_EMPLOY_TO_BIRTH_RATIO"], (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) > (-1.0))*1.) )))), -1.0 ))), ((data["DAYS_EMPLOYED"])))) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))) 
    v["i114"] = 0.098000*np.tanh(np.where(((data["DAYS_BIRTH"]) - (-2.0))>0, np.where((((1.42633831501007080)) - (data["DAYS_BIRTH"]))>0, ((((-1.0*(((1.42633831501007080))))) > (data["DAYS_BIRTH"]))*1.), -2.0 ), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] )) 
    v["i115"] = 0.095020*np.tanh((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((((np.tanh((data["ACTIVE_DAYS_CREDIT_MEAN"]))) + (data["ACTIVE_DAYS_CREDIT_MEAN"]))/2.0))))) > ((((np.where((-1.0*((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))>0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"], data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] )) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)))*1.)) 
    v["i116"] = 0.099950*np.tanh(np.where(data["EXT_SOURCE_3"]<0, np.where(data["NEW_SOURCES_PROD"] < -99998, np.where(((0.636620) + (data["EXT_SOURCE_1"]))<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["EXT_SOURCE_3"] ), data["EXT_SOURCE_3"] ), ((2.0) + (data["EXT_SOURCE_1"])) )) 
    v["i117"] = 0.089997*np.tanh((((np.minimum(((data["AMT_INCOME_TOTAL"])), ((((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, np.tanh((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])), data["NEW_INC_PER_CHLD"] )) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))))) + ((((np.tanh((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) > (data["AMT_INCOME_TOTAL"]))*1.)))/2.0)) 
    v["i118"] = 0.099801*np.tanh(((((((((((np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], (((data["EXT_SOURCE_3"]) < (((((0.318310) / 2.0)) - (2.0))))*1.) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i119"] = 0.098299*np.tanh(np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, 0.636620, data["REG_CITY_NOT_WORK_CITY"] )<0, (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (0.636620))*1.), ((np.minimum(((data["REG_CITY_NOT_WORK_CITY"])), ((((data["ORGANIZATION_TYPE_Trade__type_2"]) * 2.0))))) * (data["REG_CITY_NOT_WORK_CITY"])) )) 
    v["i120"] = 0.097399*np.tanh(np.where(data["NEW_INC_PER_CHLD"]>0, np.where(data["NAME_INCOME_TYPE_State_servant"]<0, ((0.318310) + (data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])), (((np.tanh(((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) < (0.318310))*1.)))) > (data["NEW_INC_PER_CHLD"]))*1.) ), data["NAME_INCOME_TYPE_Maternity_leave"] )) 
    v["i121"] = 0.058000*np.tanh((((((((((data["DAYS_REGISTRATION"]) - (-1.0))) - (data["NEW_INC_PER_CHLD"]))) * (data["ORGANIZATION_TYPE_Agriculture"]))) + (np.maximum(((data["ORGANIZATION_TYPE_Realtor"])), (((((-1.0) > (((data["DAYS_REGISTRATION"]) / 2.0)))*1.))))))/2.0)) 
    v["i122"] = 0.009503*np.tanh((((((data["ORGANIZATION_TYPE_Mobile"]) + (((((((((data["AMT_INCOME_TOTAL"]) + (data["AMT_INCOME_TOTAL"]))) + (np.tanh((data["FLAG_PHONE"]))))/2.0)) + (np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["NAME_INCOME_TYPE_Unemployed"])))))/2.0)))/2.0)) - (data["NAME_INCOME_TYPE_Student"]))) 
    v["i123"] = 0.029700*np.tanh((((((data["ORGANIZATION_TYPE_Mobile"]) + (((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["ORGANIZATION_TYPE_Mobile"]))) - (data["NAME_INCOME_TYPE_Student"]))))) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((((data["FLAG_PHONE"]) + (data["NEW_INC_PER_CHLD"]))/2.0)) / 2.0)))/2.0)))/2.0)) 
    v["i124"] = 0.099896*np.tanh(np.where(((data["NEW_SOURCES_PROD"]) - (-1.0))<0, (((((-1.0) * 2.0)) < (data["NEW_SOURCES_PROD"]))*1.), (((((((data["NEW_INC_PER_CHLD"]) * 2.0)) < (data["NEW_SOURCES_PROD"]))*1.)) * (data["CC_AMT_DRAWINGS_OTHER_CURRENT_MAX"])) )) 
    v["i125"] = 0.098900*np.tanh(((np.where(data["REFUSED_DAYS_DECISION_MEAN"]<0, ((((-1.0*(((((data["ORGANIZATION_TYPE_Other"]) + (data["REFUSED_AMT_APPLICATION_MEAN"]))/2.0))))) < ((((np.maximum(((data["ORGANIZATION_TYPE_Other"])), ((data["REFUSED_AMT_APPLICATION_MEAN"])))) < (data["REFUSED_DAYS_DECISION_MEAN"]))*1.)))*1.), data["OCCUPATION_TYPE_Accountants"] )) * 2.0)) 
    v["i126"] = 0.095459*np.tanh(np.where(((np.tanh(((2.84455132484436035)))) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))>0, (((data["NAME_INCOME_TYPE_Unemployed"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))/2.0), np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) )) 
    v["i127"] = 0.097680*np.tanh(np.minimum(((np.where(data["AMT_ANNUITY"]>0, data["AMT_ANNUITY"], np.minimum(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (-1.0)))), ((data["NAME_INCOME_TYPE_Maternity_leave"]))) ))), ((((((-1.0*((data["AMT_ANNUITY"])))) + (1.570796))/2.0))))) 
    v["i128"] = 0.096500*np.tanh(((np.where(data["FLAG_EMAIL"]<0, (-1.0*((np.where(data["NAME_INCOME_TYPE_Maternity_leave"]<0, ((3.0) * ((((data["EXT_SOURCE_3"]) > (1.570796))*1.))), data["EXT_SOURCE_3"] )))), data["EXT_SOURCE_3"] )) - (data["NAME_INCOME_TYPE_Student"]))) 
    v["i129"] = 0.100000*np.tanh((((((np.maximum(((data["BURO_STATUS_X_MEAN_MEAN"])), ((np.maximum(((data["EXT_SOURCE_3"])), ((((((-1.0*((0.318310)))) > (data["EXT_SOURCE_3"]))*1.)))))))) < (0.318310))*1.)) - ((((1.570796) < (data["EXT_SOURCE_3"]))*1.)))) 
    v["i130"] = 0.087069*np.tanh(np.where(data["BURO_STATUS_0_MEAN_MEAN"]>0, np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"]<0, ((((((2.0)) > (data["BURO_STATUS_0_MEAN_MEAN"]))*1.)) + (np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]<0, data["NAME_INCOME_TYPE_Working"], data["NAME_INCOME_TYPE_Maternity_leave"] ))), data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] ), data["NAME_INCOME_TYPE_Maternity_leave"] )) 
    v["i131"] = 0.099102*np.tanh(np.where(data["ORGANIZATION_TYPE_XNA"]>0, data["REGION_POPULATION_RELATIVE"], np.minimum(((data["DAYS_EMPLOYED"])), (((((-1.0*((data["DAYS_EMPLOYED"])))) * (((((-1.0*((data["DAYS_EMPLOYED"])))) + (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))/2.0)))))) )) 
    v["i132"] = 0.090200*np.tanh(np.where(np.minimum(((data["DAYS_REGISTRATION"])), ((data["DAYS_BIRTH"])))>0, (-1.0*((((data["NEW_INC_BY_ORG"]) / 2.0)))), np.where(data["DAYS_REGISTRATION"]<0, ((data["DAYS_BIRTH"]) * (data["NAME_INCOME_TYPE_State_servant"])), ((data["NAME_INCOME_TYPE_State_servant"]) / 2.0) ) )) 
    v["i133"] = 0.099230*np.tanh(((((((np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (0.636620))))) + (0.636620))/2.0)) > (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_2"]))/2.0)) > ((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))*1.)))*1.)) 
    v["i134"] = 0.095024*np.tanh(np.where(((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) < (np.tanh((data["ORGANIZATION_TYPE_Business_Entity_Type_2"]))))*1.)) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)>0, data["ORGANIZATION_TYPE_Kindergarten"], ((np.tanh((data["ORGANIZATION_TYPE_Business_Entity_Type_2"]))) - (((data["ORGANIZATION_TYPE_Kindergarten"]) / 2.0))) )) 
    v["i135"] = 0.099960*np.tanh(((((np.where((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))>0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NAME_INCOME_TYPE_Working"]))*1.), np.tanh((data["NAME_INCOME_TYPE_Working"])) )) + (np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((-1.0*((data["NAME_INCOME_TYPE_Working"]))))))))) * 2.0)) 
    v["i136"] = 0.097803*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]<0, np.maximum(((((data["ORGANIZATION_TYPE_Legal_Services"]) - (data["NAME_CONTRACT_TYPE_Revolving_loans"])))), ((data["ORGANIZATION_TYPE_Legal_Services"]))), np.maximum((((((data["CLOSED_AMT_ANNUITY_MEAN"]) > (data["ORGANIZATION_TYPE_Legal_Services"]))*1.))), (((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (3.141593))*1.)))) )) 
    v["i137"] = 0.099961*np.tanh((((((data["ORGANIZATION_TYPE_Mobile"]) - (data["NAME_INCOME_TYPE_Student"]))) + ((((((data["OCCUPATION_TYPE_Secretaries"]) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) + (((data["ORGANIZATION_TYPE_Mobile"]) * (np.where(data["OCCUPATION_TYPE_Core_staff"]>0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"], data["ORGANIZATION_TYPE_Mobile"] )))))/2.0)))/2.0)) 
    v["i138"] = 0.099620*np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, np.minimum(((data["AMT_ANNUITY"])), ((0.0))), np.where(data["ORGANIZATION_TYPE_Kindergarten"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((data["AMT_ANNUITY"]) < (((-2.0) - (np.tanh((data["AMT_ANNUITY"]))))))*1.) ) )) 
    v["i139"] = 0.093062*np.tanh(((np.where(data["NAME_INCOME_TYPE_Working"]<0, np.where(data["AMT_CREDIT"]>0, data["NAME_CONTRACT_TYPE_Cash_loans"], 0.636620 ), (-1.0*(((((data["AMT_CREDIT"]) > (np.minimum(((np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((0.636620)))))*1.)))) )) / 2.0)) 
    v["i140"] = 0.099801*np.tanh(((((np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (-1.0))*1.)))) - (((((((((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0)) / 2.0)) > (data["NAME_CONTRACT_TYPE_Cash_loans"]))*1.)))) * 2.0)) 
    v["i141"] = 0.089801*np.tanh(np.maximum(((data["BURO_STATUS_4_MEAN_MEAN"])), ((np.where((((2.0) + (data["NEW_DOC_IND_KURT"]))/2.0)>0, np.minimum(((data["NEW_DOC_IND_KURT"])), ((0.0))), ((((-1.0*((data["BURO_STATUS_4_MEAN_MEAN"])))) > (data["NAME_CONTRACT_TYPE_Cash_loans"]))*1.) ))))) 
    v["i142"] = 0.062967*np.tanh(np.where(data["BURO_STATUS_X_MEAN_MEAN"]>0, np.where(((data["BURO_STATUS_4_MEAN_MEAN"]) + (data["BURO_STATUS_X_MEAN_MEAN"]))>0, ((data["BURO_STATUS_4_MEAN_MEAN"]) - (((np.tanh((data["BURO_STATUS_X_MEAN_MEAN"]))) + (data["BURO_MONTHS_BALANCE_MIN_MIN"])))), data["BURO_MONTHS_BALANCE_MIN_MIN"] ), data["NAME_INCOME_TYPE_Maternity_leave"] )) 
    v["i143"] = 0.099749*np.tanh(np.where(data["INSTAL_PAYMENT_DIFF_VAR"] < -99998, (((data["FLAG_PHONE"]) < (0.636620))*1.), np.where(data["INSTAL_PAYMENT_DIFF_VAR"]<0, (((0.636620) < (data["INSTAL_AMT_PAYMENT_STD"]))*1.), np.where(data["INSTAL_AMT_PAYMENT_STD"]<0, 1.0, -2.0 ) ) )) 
    v["i144"] = 0.099950*np.tanh(np.where((-1.0*((np.where(data["POS_SK_DPD_DEF_MAX"]<0, data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"], (((-1.0*((data["POS_SK_DPD_MEAN"])))) + (0.318310)) ))))<0, (10.0), (-1.0*((0.318310))) )) 
    v["i145"] = 0.099710*np.tanh((((((data["ORGANIZATION_TYPE_Other"]) * (np.maximum(((np.maximum(((data["NAME_INCOME_TYPE_Student"])), ((data["NAME_INCOME_TYPE_Commercial_associate"]))))), ((data["FLAG_PHONE"])))))) + ((((np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["NAME_INCOME_TYPE_Commercial_associate"])))) < (data["FLAG_PHONE"]))*1.)))/2.0)) 
    v["i146"] = 0.099550*np.tanh(((data["REFUSED_DAYS_DECISION_MEAN"]) * (((3.0) * ((((((np.tanh(((((data["REFUSED_DAYS_DECISION_MAX"]) > (0.318310))*1.)))) > (data["REFUSED_DAYS_DECISION_MEAN"]))*1.)) - ((((0.636620) > (data["REFUSED_DAYS_DECISION_MAX"]))*1.)))))))) 
    v["i147"] = 0.099951*np.tanh(((np.tanh((((((((data["AMT_ANNUITY"]) * ((13.30679225921630859)))) * 2.0)) * 2.0)))) - (np.where(data["NAME_INCOME_TYPE_Student"] < -99998, data["EXT_SOURCE_2"], ((((data["EXT_SOURCE_2"]) + (data["EXT_SOURCE_2"]))) * 2.0) )))) 
    v["i148"] = 0.100000*np.tanh(((((((data["HOUR_APPR_PROCESS_START"]) > (data["REGION_RATING_CLIENT"]))*1.)) + (np.minimum(((np.where(data["LIVINGAREA_AVG"] < -99998, np.tanh((((data["HOUR_APPR_PROCESS_START"]) * 2.0))), (-1.0*((data["REGION_RATING_CLIENT"]))) ))), ((0.0)))))/2.0)) 
    v["i149"] = 0.093000*np.tanh(np.where((((1.570796) + (np.where(data["EXT_SOURCE_3"] < -99998, data["EXT_SOURCE_3"], data["EXT_SOURCE_1"] )))/2.0)<0, data["ORGANIZATION_TYPE_Advertising"], np.where((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (1.570796))/2.0)<0, data["EXT_SOURCE_1"], data["EXT_SOURCE_3"] ) )) 
    v["i150"] = 0.096966*np.tanh((((np.minimum(((data["EXT_SOURCE_1"])), ((np.where(data["HOUR_APPR_PROCESS_START"]>0, data["EXT_SOURCE_3"], data["NAME_INCOME_TYPE_Maternity_leave"] ))))) > ((((((0.318310) * (np.minimum(((-2.0)), ((data["EXT_SOURCE_3"])))))) + (data["EXT_SOURCE_1"]))/2.0)))*1.)) 
    v["i151"] = 0.094980*np.tanh((((((6.0)) + (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))) * (((((6.0)) < (np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]<0, ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) - (-2.0))) * 2.0), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] )))*1.)))) 
    v["i152"] = 0.099961*np.tanh((((((((3.0) < ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (((((((np.tanh((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))) + (1.0))/2.0)) + (np.maximum(((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"])), ((data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"])))))/2.0)))/2.0)))*1.)) * 2.0)) * 2.0)) 
    v["i153"] = 0.093200*np.tanh((((-1.0*((((((((((data["HOUR_APPR_PROCESS_START"]) / 2.0)) + (data["HOUR_APPR_PROCESS_START"]))/2.0)) > ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) - (0.318310))))*1.)))*1.))))) / 2.0)) 
    v["i154"] = 0.100000*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(((((-1.0*((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) < ((0.95379376411437988)))*1.)>0, data["NAME_INCOME_TYPE_Maternity_leave"], np.where(data["DAYS_BIRTH"]>0, 0.636620, (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["DAYS_BIRTH"]))/2.0) ) )))) 
    v["i155"] = 0.099949*np.tanh(((((data["AMT_INCOME_TOTAL"]) * 2.0)) * (np.maximum(((np.minimum(((((data["NEW_DOC_IND_KURT"]) * (data["NEW_DOC_IND_KURT"])))), ((data["NEW_DOC_IND_KURT"]))))), ((((data["BURO_STATUS_nan_MEAN_MEAN"]) - (((data["NEW_DOC_IND_KURT"]) * 2.0))))))))) 
    v["i156"] = 0.099200*np.tanh(((((data["ORGANIZATION_TYPE_Realtor"]) + (3.0))) * (((data["ORGANIZATION_TYPE_Realtor"]) + ((((data["ORGANIZATION_TYPE_Construction"]) > (np.where(data["EXT_SOURCE_3"] < -99998, data["ORGANIZATION_TYPE_Construction"], (((data["EXT_SOURCE_3"]) + (1.570796))/2.0) )))*1.)))))) 
    v["i157"] = 0.099950*np.tanh(((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) * (np.maximum(((data["BURO_STATUS_4_MEAN_MEAN"])), (((((((data["BURO_STATUS_0_MEAN_MEAN"]) > (2.0))*1.)) * (data["BURO_MONTHS_BALANCE_MIN_MIN"])))))))) * 2.0)) 
    v["i158"] = 0.082050*np.tanh(((data["NEW_SCORES_STD"]) * ((((((((data["NEW_SCORES_STD"]) > (((((data["NAME_TYPE_SUITE_Group_of_people"]) + (3.0))) - (0.318310))))*1.)) - (data["NAME_TYPE_SUITE_Group_of_people"]))) - (0.318310))))) 
    v["i159"] = 0.099501*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((((((data["ORGANIZATION_TYPE_Telecom"]) + (((data["ORGANIZATION_TYPE_Telecom"]) * 2.0)))/2.0)) > ((((((((data["ORGANIZATION_TYPE_Telecom"]) * 2.0)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.) )) 
    v["i160"] = 0.079550*np.tanh((-1.0*(((((data["REFUSED_AMT_APPLICATION_MEAN"]) > ((((((data["REFUSED_DAYS_DECISION_MAX"]) > (((((data["REFUSED_HOUR_APPR_PROCESS_START_MAX"]) - (data["REFUSED_AMT_APPLICATION_MEAN"]))) - (((0.318310) * 2.0)))))*1.)) * (data["REFUSED_AMT_CREDIT_MEAN"]))))*1.))))) 
    v["i161"] = 0.070000*np.tanh((((((np.maximum(((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) - (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))), ((np.maximum(((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) - (data["BURO_MONTHS_BALANCE_MAX_MAX"])))), ((np.minimum(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])), ((data["BURO_MONTHS_BALANCE_MAX_MAX"])))))))))) * 2.0)) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))*1.)) 
    v["i162"] = 0.098891*np.tanh(np.where((((2.0) < (np.maximum(((data["BURO_STATUS_2_MEAN_MEAN"])), ((data["CLOSED_MONTHS_BALANCE_MIN_MIN"])))))*1.)>0, 2.0, (-1.0*(((((data["BURO_STATUS_2_MEAN_MEAN"]) > ((-1.0*(((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (0.318310))*1.))))))*1.)))) )) 
    v["i163"] = 0.099819*np.tanh(np.where(data["CLOSED_AMT_ANNUITY_MAX"]>0, (((data["AMT_INCOME_TOTAL"]) > (0.318310))*1.), (((-2.0) > (np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["EXT_SOURCE_3"]) - (data["NAME_INCOME_TYPE_Maternity_leave"])) )))*1.) )) 
    v["i164"] = 0.097050*np.tanh((-1.0*(((((((np.maximum(((0.318310)), ((((((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) * (0.636620))) > (((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.))))) 
    v["i165"] = 0.099981*np.tanh((-1.0*((((data["NAME_TYPE_SUITE_Group_of_people"]) + (((data["ORGANIZATION_TYPE_Trade__type_6"]) + (((data["ORGANIZATION_TYPE_Trade__type_4"]) + (np.where(data["CLOSED_AMT_ANNUITY_MEAN"] < -99998, 0.0, np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["ORGANIZATION_TYPE_Trade__type_6"], data["FONDKAPREMONT_MODE_reg_oper_spec_account"] ) ))))))))))) 
    v["i166"] = 0.022300*np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NAME_EDUCATION_TYPE_Higher_education"]))) > (np.maximum(((((data["NAME_EDUCATION_TYPE_Higher_education"]) / 2.0))), ((np.maximum(((np.tanh((data["REG_CITY_NOT_WORK_CITY"])))), ((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))))))))*1.)) 
    v["i167"] = 0.087950*np.tanh((((np.where(data["WALLSMATERIAL_MODE_Panel"]<0, np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["NAME_INCOME_TYPE_Student"]))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0) )) < (np.minimum(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))), (((-1.0*((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)))))))))*1.)) 
    v["i168"] = 0.069600*np.tanh((((np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]>0, data["ORGANIZATION_TYPE_Industry__type_4"], data["NEW_ANNUITY_TO_INCOME_RATIO"] )) + ((((-1.0*(((((np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]<0, (5.13249444961547852), data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] )) < (data["AMT_ANNUITY"]))*1.))))) * (data["AMT_ANNUITY"]))))/2.0)) 
    v["i169"] = 0.057497*np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - ((((-1.0) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))<0, ((-1.0) * (1.570796)), 1.570796 )))*1.)) * 2.0)) * 2.0)) 
    v["i170"] = 0.099700*np.tanh(((((((((np.where(np.maximum(((data["WALLSMATERIAL_MODE_Panel"])), ((data["AMT_INCOME_TOTAL"])))>0, (-1.0*((data["AMT_INCOME_TOTAL"]))), (-1.0*((data["FONDKAPREMONT_MODE_reg_oper_account"]))) )) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * (data["WALLSMATERIAL_MODE_Panel"]))) 
    v["i171"] = 0.098400*np.tanh(np.where(((-1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))<0, (((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((-1.0*(((((0.0) + (3.0))/2.0))))))*1.)) * 2.0)) * 2.0)) * 2.0), data["WALLSMATERIAL_MODE_Stone__brick"] )) 
    v["i172"] = 0.099968*np.tanh(((np.minimum((((((((((0.318310) < ((((0.318310) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)))*1.)) * (2.0))) - (data["NONLIVINGAREA_MODE"])))), (((((data["NONLIVINGAREA_AVG"]) > (data["NONLIVINGAREA_MODE"]))*1.))))) * 2.0)) 
    v["i173"] = 0.090999*np.tanh(np.where(data["NONLIVINGAREA_AVG"]<0, (((((((((data["AMT_CREDIT"]) > ((2.85303425788879395)))*1.)) * 2.0)) * 2.0)) * (data["NEW_CREDIT_TO_INCOME_RATIO"])), ((((((data["NAME_INCOME_TYPE_Student"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)) > (data["AMT_CREDIT"]))*1.) )) 
    v["i174"] = 0.099500*np.tanh((((-1.0*(((((data["YEARS_BUILD_MEDI"]) > ((((np.where(data["YEARS_BUILD_MEDI"]>0, (((1.0) > (((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (1.570796))))*1.), data["LIVINGAPARTMENTS_MODE"] )) + (1.570796))/2.0)))*1.))))) * 2.0)) 
    v["i175"] = 0.099504*np.tanh(np.where(data["NONLIVINGAREA_MEDI"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], np.where(((data["AMT_CREDIT"]) - (data["NONLIVINGAREA_MEDI"]))<0, data["NAME_INCOME_TYPE_Maternity_leave"], (((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (((3.0) / 2.0)))*1.)) * 2.0)) * 2.0) ) )) 
    v["i176"] = 0.094380*np.tanh(((((data["LIVINGAPARTMENTS_MODE"]) * ((((data["TOTALAREA_MODE"]) < ((3.0)))*1.)))) * ((((np.minimum(((data["TOTALAREA_MODE"])), ((1.570796)))) < (np.where(data["NONLIVINGAREA_MODE"] < -99998, data["TOTALAREA_MODE"], data["LIVINGAPARTMENTS_AVG"] )))*1.)))) 
    v["i177"] = 0.099900*np.tanh((((-1.0*(((((data["TOTALAREA_MODE"]) > (np.where(np.where(data["LIVINGAPARTMENTS_MODE"] < -99998, data["TOTALAREA_MODE"], (((((data["EMERGENCYSTATE_MODE_No"]) * 2.0)) < (data["NONLIVINGAREA_MEDI"]))*1.) )>0, data["EMERGENCYSTATE_MODE_No"], 3.0 )))*1.))))) * 2.0)) 
    v["i178"] = 0.099700*np.tanh((((((((((data["AMT_INCOME_TOTAL"]) > ((((((((5.40422677993774414)) + (np.where(data["TOTALAREA_MODE"] < -99998, (4.0), data["NONLIVINGAREA_MEDI"] )))/2.0)) + (data["AMT_INCOME_TOTAL"]))/2.0)))*1.)) * (data["AMT_INCOME_TOTAL"]))) * 2.0)) * 2.0)) 
    v["i179"] = 0.050498*np.tanh(((np.where(np.minimum(((data["LIVINGAPARTMENTS_MEDI"])), ((data["NONLIVINGAREA_MODE"]))) < -99998, (((-1.0) < (data["LIVINGAPARTMENTS_MEDI"]))*1.), data["NONLIVINGAPARTMENTS_MEDI"] )) * ((((data["NONLIVINGAREA_MODE"]) < ((((data["NONLIVINGAREA_AVG"]) > (data["LIVINGAPARTMENTS_MODE"]))*1.)))*1.)))) 
    v["i180"] = 0.019700*np.tanh((((((((((np.minimum(((data["NONLIVINGAREA_MODE"])), ((1.570796)))) + (3.0))) < (data["LIVINGAPARTMENTS_MEDI"]))*1.)) * (((((1.570796) + (data["LIVINGAPARTMENTS_MEDI"]))) + (data["LIVINGAPARTMENTS_MODE"]))))) * 2.0)) 
    v["i181"] = 0.099496*np.tanh((((np.maximum(((np.where(((((((0.318310) < (data["LIVINGAPARTMENTS_AVG"]))*1.)) > (data["LIVINGAPARTMENTS_AVG"]))*1.)>0, data["NONLIVINGAREA_AVG"], -2.0 ))), ((((data["LIVINGAPARTMENTS_AVG"]) - (data["NONLIVINGAREA_MODE"])))))) > (1.570796))*1.)) 
    v["i182"] = 0.078968*np.tanh(((np.where((((data["LIVINGAPARTMENTS_AVG"]) > (data["LIVINGAPARTMENTS_MEDI"]))*1.)>0, (((((((data["LIVINGAPARTMENTS_MEDI"]) > (data["LIVINGAPARTMENTS_MODE"]))*1.)) - (data["LIVINGAPARTMENTS_MODE"]))) - (3.0)), data["LIVINGAPARTMENTS_MEDI"] )) - (data["LIVINGAPARTMENTS_MODE"]))) 
    v["i183"] = 0.098001*np.tanh((((((data["NONLIVINGAREA_MEDI"]) < (((data["NAME_INCOME_TYPE_Student"]) + (((data["NAME_INCOME_TYPE_Student"]) + (data["NONLIVINGAREA_AVG"]))))))*1.)) - ((((np.maximum(((data["NONLIVINGAREA_AVG"])), ((3.141593)))) < (data["NONLIVINGAREA_MEDI"]))*1.)))) 
    v["i184"] = 0.048400*np.tanh((((((((np.minimum(((np.minimum(((data["NONLIVINGAREA_AVG"])), (((-1.0*((data["NONLIVINGAREA_MODE"])))))))), ((((((np.tanh((data["EMERGENCYSTATE_MODE_No"]))) * 2.0)) / 2.0))))) * (3.141593))) > (data["NONLIVINGAREA_MEDI"]))*1.)) * 2.0)) 
    v["i185"] = 0.088029*np.tanh((-1.0*(((((((data["NONLIVINGAPARTMENTS_AVG"]) < (data["LIVINGAPARTMENTS_MODE"]))*1.)) * ((((((((data["NONLIVINGAPARTMENTS_AVG"]) + (data["NONLIVINGAREA_MODE"]))/2.0)) * (data["NONLIVINGAREA_MODE"]))) - (data["NONLIVINGAREA_MODE"])))))))) 
    v["i186"] = 0.099699*np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * ((((((data["FONDKAPREMONT_MODE_not_specified"]) + (data["FONDKAPREMONT_MODE_not_specified"]))) > ((((data["FONDKAPREMONT_MODE_not_specified"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (np.tanh((-2.0))))))/2.0)))*1.)))) 
    v["i187"] = 0.053159*np.tanh(((np.where((((data["LIVINGAPARTMENTS_AVG"]) > (data["LIVINGAPARTMENTS_MEDI"]))*1.)>0, ((data["LIVINGAPARTMENTS_MODE"]) - ((5.28531932830810547))), (((-1.0*((np.tanh((np.tanh((data["NONLIVINGAREA_MODE"])))))))) / 2.0) )) / 2.0)) 
    v["i188"] = 0.097900*np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((-1.0*((((np.where((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) / 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)>0, data["NAME_INCOME_TYPE_Student"], ((-1.0) / 2.0) )) - ((-1.0*((1.570796))))))))))*1.)) 
    v["i189"] = 0.085203*np.tanh(np.where(data["DAYS_BIRTH"]<0, np.where(data["AMT_ANNUITY"]>0, ((((data["FONDKAPREMONT_MODE_not_specified"]) * (data["AMT_ANNUITY"]))) * (data["AMT_ANNUITY"])), (((data["DAYS_BIRTH"]) > (((data["FONDKAPREMONT_MODE_not_specified"]) / 2.0)))*1.) ), data["ORGANIZATION_TYPE_Legal_Services"] )) 
    v["i190"] = 0.100000*np.tanh((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (0.636620))*1.)) * 2.0)) - (0.318310))) * ((-1.0*((np.tanh((np.minimum(((data["DAYS_BIRTH"])), (((((data["DAYS_BIRTH"]) > (0.636620))*1.)))))))))))) 
    v["i191"] = 0.099503*np.tanh((-1.0*((((((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["EXT_SOURCE_3"], 1.570796 )) + (data["AMT_ANNUITY"]))/2.0)) > (2.0))*1.)) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (5.0), -2.0 ))))))) 
    v["i192"] = 0.099943*np.tanh(np.where(((0.636620) - (data["EXT_SOURCE_3"]))<0, data["OCCUPATION_TYPE_Accountants"], np.where(data["EXT_SOURCE_3"] < -99998, np.where(data["AMT_INCOME_TOTAL"]<0, data["AMT_INCOME_TOTAL"], (((data["AMT_ANNUITY"]) < (data["ORGANIZATION_TYPE_Advertising"]))*1.) ), data["ORGANIZATION_TYPE_Advertising"] ) )) 
    v["i193"] = 0.099000*np.tanh((-1.0*((((((((((((((((((-1.0*((data["NAME_EDUCATION_TYPE_Academic_degree"])))) - (data["ORGANIZATION_TYPE_Transport__type_1"]))) * 2.0)) - (-2.0))) - (data["NAME_INCOME_TYPE_Maternity_leave"]))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * 2.0)) * 2.0))))) 
    v["i194"] = 0.099590*np.tanh(np.where(data["WALLSMATERIAL_MODE_Others"]<0, ((((-1.0*((data["WALLSMATERIAL_MODE_Others"])))) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((((((-1.0*((data["WALLSMATERIAL_MODE_Others"])))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * 2.0)))*1.)))*1.), data["AMT_ANNUITY"] )) 
    v["i195"] = 0.070396*np.tanh((((((1.570796) * ((((((data["DAYS_BIRTH"]) + (1.570796))) > (((3.0) / 2.0)))*1.)))) > (np.maximum(((((data["DAYS_BIRTH"]) + (1.570796)))), ((data["CLOSED_AMT_ANNUITY_MEAN"])))))*1.)) 
    v["i196"] = 0.084510*np.tanh(np.where(np.maximum(((data["OCCUPATION_TYPE_Sales_staff"])), ((((data["CLOSED_AMT_ANNUITY_MAX"]) / 2.0))))<0, data["ORGANIZATION_TYPE_Trade__type_2"], np.maximum(((((data["AMT_ANNUITY"]) * 2.0))), ((data["ORGANIZATION_TYPE_Trade__type_2"]))) )) 
    v["i197"] = 0.098240*np.tanh(((((((data["WALLSMATERIAL_MODE_Others"]) + (np.where(data["EXT_SOURCE_3"]>0, np.maximum(((data["ORGANIZATION_TYPE_Housing"])), ((data["WALLSMATERIAL_MODE_Others"]))), (((data["ORGANIZATION_TYPE_Housing"]) < (data["EXT_SOURCE_3"]))*1.) )))) * (((data["AMT_ANNUITY"]) * 2.0)))) / 2.0)) 
    v["i198"] = 0.099946*np.tanh(((data["WALLSMATERIAL_MODE_Others"]) + ((((data["NAME_INCOME_TYPE_Student"]) < (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["WALLSMATERIAL_MODE_Others"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["ORGANIZATION_TYPE_Industry__type_3"])) )))*1.)))) 
    v["i199"] = 0.089500*np.tanh((((((((((3.0) / 2.0)) < ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))*1.)) * 2.0)) - (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["ORGANIZATION_TYPE_Industry__type_12"], (((-1.0*((data["ORGANIZATION_TYPE_Industry__type_12"])))) - (data["ORGANIZATION_TYPE_Industry__type_12"])) )))) 
    v["i200"] = 0.079602*np.tanh(np.minimum(((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((((((((data["AMT_ANNUITY"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) < (((((((data["AMT_ANNUITY"]) > (data["EXT_SOURCE_3"]))*1.)) < (((data["AMT_ANNUITY"]) * 2.0)))*1.)))*1.))))) 
    v["i201"] = 0.090045*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, ((-1.0) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((2.0) + (data["WALLSMATERIAL_MODE_Mixed"]))))*1.))), (-1.0*((data["ORGANIZATION_TYPE_Industry__type_12"]))) )) 
    v["i202"] = 0.097199*np.tanh((-1.0*((np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((0.636620) - ((((((np.minimum(((np.tanh((np.tanh((0.318310)))))), ((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))) * 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.))) ))))) 
    v["i203"] = 0.068270*np.tanh((((((np.where(data["CC_NAME_CONTRACT_STATUS_Active_SUM"]<0, (-1.0*((data["CC_NAME_CONTRACT_STATUS_Refused_MEAN"]))), np.minimum(((data["CC_SK_DPD_DEF_MEAN"])), ((data["CC_NAME_CONTRACT_STATUS_Active_SUM"]))) )) < (np.where(data["CC_COUNT"]>0, data["CC_NAME_CONTRACT_STATUS_Refused_MEAN"], data["CC_SK_DPD_DEF_MEAN"] )))*1.)) * 2.0)) 
    v["i204"] = 0.098990*np.tanh(np.where(data["CLOSED_AMT_ANNUITY_MEAN"] < -99998, (((-1.0*(((((((data["EXT_SOURCE_3"]) < (((((0.636620) / 2.0)) + (data["AMT_ANNUITY"]))))*1.)) / 2.0))))) / 2.0), (((0.636620) < (data["AMT_ANNUITY"]))*1.) )) 
    v["i205"] = 0.030500*np.tanh((((-1.0*((np.where(data["BURO_STATUS_4_MEAN_MEAN"]>0, data["BURO_STATUS_C_MEAN_MEAN"], (((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (((1.0) - (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"]>0, data["AMT_CREDIT"], data["BURO_STATUS_4_MEAN_MEAN"] )))))*1.) ))))) * 2.0)) 
    v["i206"] = 0.068540*np.tanh(np.where((((((((-1.0) + ((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))/2.0)))/2.0)) - (np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))))) * 2.0)<0, data["CLOSED_MONTHS_BALANCE_MIN_MIN"], data["NAME_INCOME_TYPE_Maternity_leave"] )) 
    v["i207"] = 0.099171*np.tanh(((((((((-1.0) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.tanh((np.tanh((-1.0))))))/2.0)))*1.)) / 2.0)) > ((((3.141593) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))/2.0)))*1.)) 
    v["i208"] = 0.030500*np.tanh((-1.0*(((((((np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]<0, data["BURO_STATUS_C_MEAN_MEAN"], np.tanh((np.maximum(((data["CLOSED_MONTHS_BALANCE_MAX_MAX"])), ((data["BURO_STATUS_X_MEAN_MEAN"]))))) )) > (np.minimum(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])), ((data["CLOSED_MONTHS_BALANCE_MAX_MAX"])))))*1.)) / 2.0))))) 
    v["i209"] = 0.091100*np.tanh(np.where(data["NEW_DOC_IND_KURT"]>0, (-1.0*(((((((2.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < (data["NEW_DOC_IND_KURT"]))*1.)))), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) 
    v["i210"] = 0.069001*np.tanh(np.tanh((np.tanh(((((data["DAYS_BIRTH"]) < (((np.tanh((((((0.37940034270286560)) > (data["AMT_ANNUITY"]))*1.)))) - (np.where(data["CC_NAME_CONTRACT_STATUS_Refused_SUM"] < -99998, (2.0), data["DAYS_BIRTH"] )))))*1.)))))) 
    v["i211"] = 0.099939*np.tanh((((np.minimum(((data["CNT_CHILDREN"])), ((((((data["CNT_CHILDREN"]) * (((data["CNT_CHILDREN"]) * (data["AMT_ANNUITY"]))))) + ((((0.318310) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))*1.))))))) + (0.318310))/2.0)) 
    v["i212"] = 0.099979*np.tanh((((((((data["NAME_INCOME_TYPE_Maternity_leave"]) * (data["NAME_CONTRACT_TYPE_Cash_loans"]))) + (np.minimum(((((np.where(data["NEW_DOC_IND_KURT"]<0, data["NAME_CONTRACT_TYPE_Revolving_loans"], data["NAME_CONTRACT_TYPE_Cash_loans"] )) / 2.0))), (((((data["NEW_DOC_IND_KURT"]) < (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.))))))/2.0)) / 2.0)) 
    v["i213"] = 0.096700*np.tanh(((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * (np.where(data["INSTAL_PAYMENT_DIFF_STD"] < -99998, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], (((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) < (np.tanh((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))))))*1.) )))))) 
    v["i214"] = 0.099035*np.tanh((-1.0*(((((((0.318310) < ((((data["DAYS_BIRTH"]) + ((((data["DAYS_BIRTH"]) < ((((((data["LIVE_REGION_NOT_WORK_REGION"]) + ((((data["DAYS_BIRTH"]) + (data["LIVE_REGION_NOT_WORK_REGION"]))/2.0)))/2.0)) / 2.0)))*1.)))/2.0)))*1.)) / 2.0))))) 
    v["i215"] = 0.095997*np.tanh((((np.where(data["FLAG_WORK_PHONE"]>0, data["DAYS_BIRTH"], (((np.tanh((data["DAYS_BIRTH"]))) > ((((0.318310) + (data["DAYS_BIRTH"]))/2.0)))*1.) )) + ((((((data["DAYS_BIRTH"]) < (data["NEW_LIVE_IND_SUM"]))*1.)) / 2.0)))/2.0)) 
    v["i216"] = 0.098960*np.tanh((((0.636620) + (np.minimum(((-1.0)), ((((np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])) )) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))))/2.0)) 
    v["i217"] = 0.099400*np.tanh((-1.0*(((((((((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)) + (data["AMT_ANNUITY"]))/2.0)) < (np.minimum((((-1.0*(((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))*1.)))))), ((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))))*1.))))) 
    v["i218"] = 0.096740*np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((((((6.85217618942260742)) / 2.0)) + (np.where(data["AMT_ANNUITY"]>0, np.maximum(((data["AMT_ANNUITY"])), ((0.636620))), ((data["FLAG_CONT_MOBILE"]) * 2.0) )))/2.0)))*1.)) 
    v["i219"] = 0.099379*np.tanh((((((3.0) < (((np.where(data["NONLIVINGAREA_MEDI"]>0, data["NONLIVINGAREA_MODE"], (-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) )) * 2.0)))*1.)) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((data["AMT_ANNUITY"]) * 2.0)))*1.)))) 
    v["i220"] = 0.100000*np.tanh(((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0)) * 2.0)) * ((((data["AMT_ANNUITY"]) > ((((((data["NONLIVINGAREA_AVG"]) < ((-1.0*((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))))))*1.)) + (1.570796))))*1.)))) 
    v["i221"] = 0.099400*np.tanh((-1.0*(((((((((((5.76207304000854492)) > (data["NONLIVINGAREA_AVG"]))*1.)) < (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, 1.0, ((np.tanh(((((-1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)))) * 2.0) )))*1.)) * 2.0))))) 
    v["i222"] = 0.0*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(data["NONLIVINGAREA_MEDI"]<0, (((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.), ((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))*1.)) > (((data["NONLIVINGAREA_MODE"]) * 2.0)))*1.) )))) 
    v["i223"] = 0.097100*np.tanh((((((((((((np.where(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NONLIVINGAREA_MODE"] )>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NONLIVINGAREA_AVG"] )) > (2.0))*1.)) * 2.0)) * 2.0)) * 2.0)) - (data["NAME_INCOME_TYPE_Student"]))) 
    v["i224"] = 0.100000*np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NONLIVINGAREA_MODE"]))/2.0)) * ((((((np.maximum((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NONLIVINGAREA_MODE"]))/2.0))), ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0))))) - (3.141593))) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) 
    v["i225"] = 0.098097*np.tanh(((((np.where(np.where(data["NONLIVINGAREA_MODE"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NEW_ANNUITY_TO_INCOME_RATIO"] )<0, data["NONLIVINGAREA_MEDI"], 0.318310 )) - (data["NONLIVINGAREA_MODE"]))) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0)))) 
    v["i226"] = 0.091960*np.tanh(np.where(((np.where((-1.0*((data["NONLIVINGAREA_MEDI"])))<0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.), data["NONLIVINGAREA_MEDI"] )) - (data["NONLIVINGAREA_MODE"]))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((data["NONLIVINGAREA_MODE"]) - (data["NONLIVINGAREA_AVG"])) )) 
    v["i227"] = 0.099197*np.tanh(((((data["AMT_ANNUITY"]) * (data["ORGANIZATION_TYPE_Housing"]))) - (np.maximum(((data["OCCUPATION_TYPE_IT_staff"])), ((np.minimum(((0.318310)), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["ORGANIZATION_TYPE_Housing"]) > (((data["AMT_ANNUITY"]) / 2.0)))*1.)))))))))))) 
    v["i228"] = 0.089500*np.tanh((((np.maximum(((3.141593)), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) < (((data["AMT_ANNUITY"]) * (((np.where(((data["AMT_ANNUITY"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))>0, 1.570796, data["NEW_ANNUITY_TO_INCOME_RATIO"] )) - (1.570796))))))*1.)) 
    v["i229"] = 0.091950*np.tanh((((((np.maximum(((((((3.0) - ((((data["AMT_ANNUITY"]) < (np.tanh((np.tanh((data["NEW_CREDIT_TO_INCOME_RATIO"]))))))*1.)))) + (data["AMT_ANNUITY"])))), ((1.570796)))) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) * 2.0)) 
    v["i230"] = 0.099930*np.tanh((-1.0*((((((((data["CNT_CHILDREN"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((((data["CNT_CHILDREN"]) + ((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)))/2.0)))*1.)))*1.))))) 
    v["i231"] = 0.038000*np.tanh((((np.where(np.minimum((((((0.636620) + (data["NEW_INC_PER_CHLD"]))/2.0))), (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["AMT_ANNUITY"]) / 2.0)))/2.0))))<0, 1.570796, 3.141593 )) < (data["AMT_ANNUITY"]))*1.)) 
    v["i232"] = 0.100000*np.tanh(((((data["NEW_INC_PER_CHLD"]) * ((((data["CNT_CHILDREN"]) > (2.0))*1.)))) + ((((((((2.0) < ((((data["CNT_CHILDREN"]) + (data["NEW_INC_PER_CHLD"]))/2.0)))*1.)) * 2.0)) * 2.0)))) 
    v["i233"] = 0.074001*np.tanh((-1.0*(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((0.636620) - ((((((5.0)) - (data["NAME_INCOME_TYPE_Student"]))) - (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, 3.0, ((data["NONLIVINGAREA_MODE"]) * 2.0) )))))))*1.))))) 
    v["i234"] = 0.079999*np.tanh(((((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["ORGANIZATION_TYPE_Hotel"], ((np.tanh((np.tanh((((((data["AMT_ANNUITY"]) + (data["NONLIVINGAREA_AVG"]))) * 2.0)))))) * (data["ORGANIZATION_TYPE_Hotel"])) )) * 2.0)) * 2.0)) 
    v["i235"] = 0.078003*np.tanh(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, ((((-1.0*((0.636620)))) > (np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.), ((data["WALLSMATERIAL_MODE_Mixed"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_1"])) )) * (0.636620))) 
    v["i236"] = 0.075100*np.tanh((((np.where((((data["AMT_ANNUITY"]) > (0.636620))*1.)>0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) )) > (3.141593))*1.)) 
    v["i237"] = 0.099651*np.tanh((-1.0*(((((1.570796) < (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NAME_INCOME_TYPE_Student"]))) + ((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))*1.)) / 2.0)) / 2.0)))))*1.))))) 
    v["i238"] = 0.044000*np.tanh((((np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)))<0, data["NEW_CREDIT_TO_INCOME_RATIO"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.) )) > ((((1.0) > (data["REG_CITY_NOT_WORK_CITY"]))*1.)))*1.)) 
    v["i239"] = 0.090029*np.tanh(((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * ((((((((((((3.141593) < (data["AMT_ANNUITY"]))*1.)) * 2.0)) * 2.0)) * 2.0)) - (data["ORGANIZATION_TYPE_Trade__type_4"]))))) * 2.0)) - (data["ORGANIZATION_TYPE_Trade__type_4"]))) 
    v["i240"] = 0.090000*np.tanh(((np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NAME_EDUCATION_TYPE_Academic_degree"]))<0, ((data["AMT_ANNUITY"]) * (((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)) * 2.0))), ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)) * 2.0) )) * (data["REG_CITY_NOT_WORK_CITY"]))) 
    v["i241"] = 0.082502*np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, 2.0, data["NAME_EDUCATION_TYPE_Academic_degree"] )) * ((((-1.0*((data["NAME_EDUCATION_TYPE_Academic_degree"])))) + ((-1.0*(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)))*1.))))))))) 
    v["i242"] = 0.081500*np.tanh(np.where(data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]>0, np.maximum((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) < (1.570796))*1.))), (((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))), ((((((3.0) + ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))/2.0)) < (data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]))*1.) )) 
    v["i243"] = 0.099150*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["HOUSETYPE_MODE_specific_housing"], ((np.where(-1.0>0, data["HOUSETYPE_MODE_specific_housing"], ((-1.0) + ((((data["DAYS_BIRTH"]) < (((3.0) / 2.0)))*1.))) )) - (data["NAME_HOUSING_TYPE_Office_apartment"])) )) 
    v["i244"] = 0.099990*np.tanh(np.where(data["OCCUPATION_TYPE_Core_staff"]>0, (((((data["DAYS_ID_PUBLISH"]) * (data["AMT_ANNUITY"]))) < (data["AMT_ANNUITY"]))*1.), (-1.0*(((((((-1.0) > ((((data["DAYS_ID_PUBLISH"]) + (data["AMT_ANNUITY"]))/2.0)))*1.)) * 2.0)))) )) 
    v["i245"] = 0.059010*np.tanh((((((((-1.0*(((((data["ORGANIZATION_TYPE_Trade__type_4"]) > ((((((((((data["ORGANIZATION_TYPE_Bank"]) / 2.0)) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)) * (data["ORGANIZATION_TYPE_Trade__type_4"]))))*1.))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i246"] = 0.099998*np.tanh((((((((((((-1.0*((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))/2.0)) > (1.570796))*1.))))) * 2.0)) - (data["NAME_INCOME_TYPE_Student"]))) * 2.0)) * 2.0)) - (data["ORGANIZATION_TYPE_Trade__type_4"]))) 
    v["i247"] = 0.050000*np.tanh(((((data["INSTAL_AMT_PAYMENT_STD"]) * ((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) > ((((data["INSTAL_AMT_PAYMENT_STD"]) < (2.0))*1.)))*1.)))) + ((((np.where(data["INSTAL_AMT_PAYMENT_STD"] < -99998, data["INSTAL_AMT_PAYMENT_STD"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) > (2.0))*1.)))) 
    v["i248"] = 0.085002*np.tanh(np.minimum(((np.minimum(((((data["INSTAL_PAYMENT_DIFF_STD"]) * (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))), ((((data["INSTAL_PAYMENT_DIFF_STD"]) - (data["INSTAL_PAYMENT_DIFF_VAR"]))))))), ((np.where(data["INSTAL_PAYMENT_PERC_STD"]<0, (-1.0*((data["INSTAL_PAYMENT_DIFF_VAR"]))), data["INSTAL_AMT_PAYMENT_STD"] ))))) 
    v["i249"] = 0.092700*np.tanh(np.maximum(((np.where(data["REGION_RATING_CLIENT_W_CITY"]<0, (((-1.0*((np.tanh((data["OCCUPATION_TYPE_Waiters_barmen_staff"])))))) * (data["REGION_RATING_CLIENT_W_CITY"])), (((data["NAME_HOUSING_TYPE_With_parents"]) + (data["OCCUPATION_TYPE_Waiters_barmen_staff"]))/2.0) ))), (((((((0.16719464957714081)) / 2.0)) / 2.0))))) 
    v["i250"] = 0.099695*np.tanh(((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) / 2.0)) > (data["ORGANIZATION_TYPE_Housing"]))*1.)) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((data["ORGANIZATION_TYPE_Housing"]) / 2.0)))*1.)))*1.)) 
    v["i251"] = 0.097400*np.tanh(np.minimum(((((data["NEW_DOC_IND_KURT"]) - (np.where(data["DAYS_BIRTH"]>0, data["DAYS_REGISTRATION"], (-1.0*((((((data["DAYS_REGISTRATION"]) * ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) * 2.0)))) ))))), ((data["NAME_CONTRACT_TYPE_Cash_loans"])))) 
    v["i252"] = 0.099195*np.tanh((((((((data["INSTAL_PAYMENT_PERC_STD"]) > (((data["INSTAL_PAYMENT_PERC_VAR"]) - ((((((0.636620) * 2.0)) < (data["DAYS_REGISTRATION"]))*1.)))))*1.)) * (((((((data["DAYS_REGISTRATION"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) 
    v["i253"] = 0.098171*np.tanh((((np.where(data["NEW_DOC_IND_KURT"]>0, 2.0, np.where(((data["NEW_DOC_IND_KURT"]) + (3.0))>0, ((data["INSTAL_AMT_PAYMENT_STD"]) * 2.0), ((0.0) - (0.318310)) ) )) < (data["INSTAL_AMT_PAYMENT_STD"]))*1.)) 
    v["i254"] = 0.097303*np.tanh(((np.where(data["INSTAL_PAYMENT_DIFF_VAR"]<0, (-1.0*((data["NAME_INCOME_TYPE_Student"]))), np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]<0, (-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))), data["AMT_ANNUITY"] ) ) )) * (-2.0))) 
    v["i255"] = 0.099660*np.tanh((((((-1.0*(((((np.tanh((1.570796))) < (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, data["INSTAL_DAYS_ENTRY_PAYMENT_STD"], (((np.minimum(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) > (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))*1.) )))*1.))))) * 2.0)) * 2.0)) 
    v["i256"] = 0.099733*np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((((0.0) < ((((1.0) + (np.maximum(((data["CLOSED_AMT_ANNUITY_MEAN"])), ((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]>0, data["INSTAL_DAYS_ENTRY_PAYMENT_STD"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] ))))))/2.0)))*1.)) * 2.0)))*1.)) * 2.0)) 
    v["i257"] = 0.099308*np.tanh(np.where(data["BURO_AMT_ANNUITY_MEAN"]<0, np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"], (((np.where(data["BURO_AMT_ANNUITY_MEAN"] < -99998, data["BURO_AMT_ANNUITY_MAX"], ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * ((5.97321271896362305))) )) > (data["BURO_AMT_ANNUITY_MEAN"]))*1.) ), data["CLOSED_AMT_ANNUITY_MAX"] )) 
    v["i258"] = 0.077200*np.tanh(((((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) + (data["NAME_INCOME_TYPE_Maternity_leave"]))/2.0)) < (data["POS_SK_DPD_MEAN"]))*1.)) + (np.where(data["BURO_AMT_ANNUITY_MEAN"] < -99998, np.tanh((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])), (((-1.0*((data["BURO_AMT_ANNUITY_MEAN"])))) - (data["POS_SK_DPD_MAX"])) )))) 
    v["i259"] = 0.098503*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(data["INSTAL_AMT_PAYMENT_STD"]<0, ((np.where(data["INSTAL_PAYMENT_DIFF_VAR"]>0, (6.0), np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["INSTAL_PAYMENT_DIFF_VAR"], data["INSTAL_PAYMENT_DIFF_STD"] ) )) / 2.0), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0) )))) 
    v["i260"] = 0.099800*np.tanh(((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - ((1.95137548446655273)))))) * ((((1.570796) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.)))) - ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((1.95137548446655273)))*1.)))) * 2.0)) 
    v["i261"] = 0.079810*np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((-1.0*((0.636620)))))<0, ((1.570796) + ((-1.0*((1.0))))), 2.0 )))*1.)) 
    v["i262"] = 0.097999*np.tanh((-1.0*((np.where((((((-2.0) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)) / 2.0)>0, (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))/2.0)) / 2.0), ((((5.0)) < (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))*1.) ))))) 
    v["i263"] = 0.098397*np.tanh(((((np.where(data["OCCUPATION_TYPE_Accountants"]>0, data["OCCUPATION_TYPE_Accountants"], ((data["ORGANIZATION_TYPE_Bank"]) * (data["OCCUPATION_TYPE_Accountants"])) )) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (((data["OCCUPATION_TYPE_Accountants"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))))) 
    v["i264"] = 0.099459*np.tanh(((((((np.maximum(((data["EXT_SOURCE_3"])), ((((2.0) - (data["EXT_SOURCE_3"])))))) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)) < ((((((2.0) - (data["EXT_SOURCE_3"]))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))*1.)) 
    v["i265"] = 0.000780*np.tanh(((((((((data["ORGANIZATION_TYPE_Mobile"]) - ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((((2.0)) - (data["NAME_INCOME_TYPE_Student"]))) - (data["ORGANIZATION_TYPE_Bank"]))))*1.)))) * 2.0)) - (data["ORGANIZATION_TYPE_Bank"]))) * 2.0)) 
    v["i266"] = 0.100000*np.tanh((((-1.0*((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["ORGANIZATION_TYPE_Mobile"])))))) - (np.maximum((((((data["ORGANIZATION_TYPE_Bank"]) + (data["ORGANIZATION_TYPE_Mobile"]))/2.0))), (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (3.0))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.))))))) 
    v["i267"] = 0.099899*np.tanh(((((((data["ORGANIZATION_TYPE_Mobile"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["FLAG_DOCUMENT_3"], data["ORGANIZATION_TYPE_Mobile"] )))*1.)) > (np.where(data["ORGANIZATION_TYPE_Mobile"]>0, data["FLAG_DOCUMENT_3"], (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((3.0) / 2.0)))/2.0) )))*1.)) 
    v["i268"] = 0.096620*np.tanh(((np.where(data["NEW_CREDIT_TO_GOODS_RATIO"]<0, -2.0, ((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (data["NEW_CREDIT_TO_GOODS_RATIO"])) )) + (np.where(((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (0.318310))<0, (-1.0*((data["AMT_ANNUITY"]))), data["AMT_ANNUITY"] )))) 
    v["i269"] = 0.099659*np.tanh(((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["AMT_CREDIT"])), data["FLAG_DOCUMENT_3"] )) * (((((((((data["FLAG_WORK_PHONE"]) - (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (data["FLAG_WORK_PHONE"]))/2.0)) + (data["NAME_CONTRACT_TYPE_Revolving_loans"]))/2.0)))) 
    v["i270"] = 0.096999*np.tanh(((((((((((((data["DAYS_BIRTH"]) * (data["AMT_ANNUITY"]))) * (data["AMT_ANNUITY"]))) * (data["AMT_ANNUITY"]))) - (0.318310))) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.)))) / 2.0)) 
    v["i271"] = 0.089701*np.tanh(np.where((((((data["REGION_POPULATION_RELATIVE"]) * 2.0)) > (1.570796))*1.)>0, -1.0, (((0.636620) + (np.minimum(((((np.tanh((data["REGION_POPULATION_RELATIVE"]))) * 2.0))), ((1.0)))))/2.0) )) 
    v["i272"] = 0.100000*np.tanh((((-1.0) > (((1.570796) * (np.where(((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (0.636620))<0, (-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))), np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"])) )))))*1.)) 
    v["i273"] = 0.094500*np.tanh(np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, ((((((data["DAYS_BIRTH"]) < (0.318310))*1.)) + (data["AMT_CREDIT"]))/2.0), ((data["DAYS_BIRTH"]) * (np.tanh((np.where(data["REGION_RATING_CLIENT"]>0, data["REGION_RATING_CLIENT"], data["OCCUPATION_TYPE_Managers"] ))))) )) 
    v["i274"] = 0.094533*np.tanh((((-1.0*((np.maximum(((data["REFUSED_DAYS_DECISION_MEAN"])), ((data["REGION_RATING_CLIENT_W_CITY"]))))))) * ((((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["NAME_CONTRACT_TYPE_Revolving_loans"]))) < (np.tanh((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (data["HOUR_APPR_PROCESS_START"]))))))*1.)))) 
    v["i275"] = 0.098000*np.tanh((((data["REFUSED_AMT_GOODS_PRICE_MAX"]) > (np.maximum(((np.where(((data["REFUSED_AMT_ANNUITY_MEAN"]) - (data["REFUSED_APP_CREDIT_PERC_MEAN"]))>0, ((((data["REFUSED_AMT_GOODS_PRICE_MAX"]) * 2.0)) + (data["REFUSED_AMT_ANNUITY_MEAN"])), data["REFUSED_AMT_ANNUITY_MEAN"] ))), ((((-1.0) / 2.0))))))*1.)) 
    v["i276"] = 0.096999*np.tanh(((np.minimum(((data["NAME_CONTRACT_TYPE_Revolving_loans"])), (((-1.0*((np.where((((0.636620) + (np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["AMT_ANNUITY"])))))/2.0)<0, data["AMT_ANNUITY"], data["NAME_CONTRACT_TYPE_Revolving_loans"] )))))))) / 2.0)) 
    v["i277"] = 0.040300*np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((((((data["FLAG_EMAIL"]) > (data["AMT_ANNUITY"]))*1.)) + (np.minimum(((((1.570796) * (-2.0)))), ((((data["FLAG_EMAIL"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))/2.0)))*1.)) 
    v["i278"] = 0.099200*np.tanh(((np.where((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (np.tanh((0.636620))))*1.)>0, data["AMT_ANNUITY"], (((((np.tanh((1.570796))) < (np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["FLAG_OWN_REALTY"])))))*1.)) / 2.0) )) * 2.0)) 
    v["i279"] = 0.082488*np.tanh(((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["AMT_ANNUITY"], (-1.0*((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * ((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > ((((-1.0*((0.636620)))) / 2.0)))*1.)) - (data["AMT_ANNUITY"]))))))) )) / 2.0)) 
    v["i280"] = 0.097030*np.tanh(np.minimum(((((data["REGION_POPULATION_RELATIVE"]) * (((data["EXT_SOURCE_3"]) / 2.0))))), ((((data["AMT_CREDIT"]) * (np.where(((data["EXT_SOURCE_3"]) / 2.0)<0, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["REGION_POPULATION_RELATIVE"]) * (data["EXT_SOURCE_3"])) ))))))) 
    v["i281"] = 0.099501*np.tanh(np.where(((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) - ((0.94349050521850586)))<0, ((np.minimum(((((((0.94349050521850586)) > (data["REGION_RATING_CLIENT"]))*1.))), ((0.318310)))) / 2.0), ((data["REGION_RATING_CLIENT"]) - (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])) )) 
    v["i282"] = 0.098896*np.tanh(((np.where(data["NEW_SOURCES_PROD"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where((((data["NEW_SOURCES_PROD"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)>0, data["NEW_SOURCES_PROD"], (((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((1.570796)))) < (data["HOUR_APPR_PROCESS_START"]))*1.) ) )) * 2.0)) 
    v["i283"] = 0.099500*np.tanh(np.maximum(((((data["NEW_SCORES_STD"]) * ((((3.141593) < (data["NEW_SCORES_STD"]))*1.))))), (((((np.tanh((np.tanh((data["NEW_CAR_TO_EMPLOY_RATIO"]))))) > (((np.maximum(((data["AMT_INCOME_TOTAL"])), ((0.318310)))) / 2.0)))*1.))))) 
    v["i284"] = 0.098199*np.tanh((((((((0.636620) * 2.0)) < (data["OWN_CAR_AGE"]))*1.)) - (np.where(((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (data["OWN_CAR_AGE"]))<0, 0.318310, ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)) / 2.0) )))) 
    v["i285"] = 0.099860*np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) > (np.maximum((((((data["NEW_INC_PER_CHLD"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.))), ((data["AMT_ANNUITY"])))))*1.)))) 
    v["i286"] = 0.079000*np.tanh((((((np.tanh((1.570796))) < (data["LIVINGAPARTMENTS_MODE"]))*1.)) * ((((((((data["LIVINGAPARTMENTS_AVG"]) / 2.0)) < (1.570796))*1.)) * (((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)) * 2.0)))))) 
    v["i287"] = 0.099896*np.tanh(((((((((1.0) + ((((0.636620) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))/2.0)) < (np.tanh(((((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)))))*1.)) * ((4.11586141586303711)))) 
    v["i288"] = 0.096495*np.tanh(np.minimum(((((data["OCCUPATION_TYPE_Laborers"]) * (((((data["AMT_ANNUITY"]) / 2.0)) / 2.0))))), ((np.where(np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["AMT_ANNUITY"])))<0, 0.0, (((data["OCCUPATION_TYPE_Laborers"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0) ))))) 
    v["i289"] = 0.062201*np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((((np.tanh((np.minimum(((data["OCCUPATION_TYPE_Managers"])), ((((data["NAME_INCOME_TYPE_Working"]) * (np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((0.318310) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))))))))))) * 2.0)) * 2.0)))*1.)) 
    v["i290"] = 0.097950*np.tanh(((((((data["AMT_ANNUITY"]) > (data["LIVINGAPARTMENTS_MODE"]))*1.)) < (np.where(((((((((data["AMT_ANNUITY"]) - (data["AMT_INCOME_TOTAL"]))) * 2.0)) / 2.0)) * 2.0)>0, data["NEW_INC_PER_CHLD"], data["AMT_INCOME_TOTAL"] )))*1.)) 
    v["i291"] = 0.097700*np.tanh((((((data["NONLIVINGAPARTMENTS_MEDI"]) > (np.where(np.minimum(((data["LIVINGAPARTMENTS_AVG"])), ((data["LIVINGAPARTMENTS_AVG"])))<0, 0.318310, (((((data["NONLIVINGAPARTMENTS_AVG"]) > (data["LIVINGAPARTMENTS_AVG"]))*1.)) + (data["LIVINGAPARTMENTS_MODE"])) )))*1.)) * 2.0)) 
    v["i292"] = 0.026250*np.tanh((((((data["COMMONAREA_MODE"]) > (3.0))*1.)) - ((((data["LIVINGAPARTMENTS_AVG"]) > (((data["NONLIVINGAPARTMENTS_AVG"]) - (((((((((data["COMMONAREA_MODE"]) < (data["LIVINGAPARTMENTS_MODE"]))*1.)) / 2.0)) < (data["NONLIVINGAPARTMENTS_AVG"]))*1.)))))*1.)))) 
    v["i293"] = 0.090030*np.tanh(((np.where(data["LIVINGAPARTMENTS_MEDI"]>0, (((((data["LIVINGAPARTMENTS_MEDI"]) < (data["NONLIVINGAPARTMENTS_AVG"]))*1.)) * (((-2.0) + (data["NONLIVINGAPARTMENTS_MODE"])))), (((data["NONLIVINGAPARTMENTS_AVG"]) > (0.318310))*1.) )) * 2.0)) 
    v["i294"] = 0.055511*np.tanh(np.minimum(((np.where(data["NONLIVINGAPARTMENTS_AVG"] < -99998, np.where(data["LIVINGAPARTMENTS_MODE"] < -99998, (((data["NONLIVINGAPARTMENTS_AVG"]) < (data["NONLIVINGAPARTMENTS_MODE"]))*1.), data["LIVINGAPARTMENTS_MEDI"] ), data["NONLIVINGAPARTMENTS_MODE"] ))), ((((((((2.32806849479675293)) < (data["NONLIVINGAPARTMENTS_MODE"]))*1.)) * 2.0))))) 
    v["i295"] = 0.039101*np.tanh(((((((((((((((data["NONLIVINGAPARTMENTS_MODE"]) * ((((((1.570796) + ((0.57018411159515381)))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i296"] = 0.073000*np.tanh((((-1.0*(((((data["NONLIVINGAPARTMENTS_MEDI"]) > (((((-1.0*((np.where(data["LIVINGAPARTMENTS_AVG"]<0, data["NONLIVINGAPARTMENTS_MEDI"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] ))))) < ((((((data["NONLIVINGAPARTMENTS_MODE"]) / 2.0)) + (data["LIVINGAPARTMENTS_MODE"]))/2.0)))*1.)))*1.))))) * 2.0)) 
    v["i297"] = 0.095100*np.tanh(np.where(data["ORGANIZATION_TYPE_School"]>0, data["OCCUPATION_TYPE_Laborers"], np.where(data["NAME_TYPE_SUITE_Other_A"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((data["ORGANIZATION_TYPE_School"]) * (np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NAME_TYPE_SUITE_Other_A"]))<0, data["ORGANIZATION_TYPE_Security_Ministries"], data["OCCUPATION_TYPE_Laborers"] ))) ) )) 
    v["i298"] = 0.094600*np.tanh(((-1.0) + ((((((data["AMT_ANNUITY"]) - (1.570796))) < ((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))) / 2.0)))*1.)))) 
    v["i299"] = 0.059201*np.tanh(((((np.where(data["FLAG_DOCUMENT_3"]<0, data["AMT_ANNUITY"], ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (data["NAME_TYPE_SUITE_Other_A"])) )) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (((data["NAME_TYPE_SUITE_Other_A"]) / 2.0)))) 
    v["i300"] = 0.090199*np.tanh(((2.0) * ((((((((data["AMT_ANNUITY"]) < (0.636620))*1.)) * (((np.where(((((2.06409978866577148)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)>0, data["NAME_INCOME_TYPE_Student"], data["AMT_ANNUITY"] )) * 2.0)))) * 2.0)))) 
    v["i301"] = 0.085101*np.tanh(((np.where(np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))>0, data["ORGANIZATION_TYPE_Industry__type_12"], ((((data["ORGANIZATION_TYPE_Industry__type_12"]) * ((-1.0*((((data["FLAG_DOCUMENT_3"]) * 2.0))))))) - (data["ORGANIZATION_TYPE_Industry__type_12"])) )) - (data["NAME_INCOME_TYPE_Student"]))) 
    v["i302"] = 0.081773*np.tanh((((3.0) < (((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, 3.141593, (((((1.570796) < (data["AMT_ANNUITY"]))*1.)) - (data["ORGANIZATION_TYPE_Industry__type_12"])) )) - (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))))*1.)) 
    v["i303"] = 0.093010*np.tanh(((((((data["REFUSED_RATE_DOWN_PAYMENT_MAX"]) > (np.tanh((np.tanh((np.tanh((2.0))))))))*1.)) < (np.where(data["REFUSED_AMT_DOWN_PAYMENT_MAX"]>0, np.where(data["REFUSED_AMT_DOWN_PAYMENT_MEAN"]>0, data["REFUSED_AMT_DOWN_PAYMENT_MEAN"], data["REFUSED_RATE_DOWN_PAYMENT_MAX"] ), data["REFUSED_AMT_DOWN_PAYMENT_MAX"] )))*1.)) 
    v["i304"] = 0.098000*np.tanh((-1.0*(((((data["REFUSED_RATE_DOWN_PAYMENT_MAX"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, np.where(data["REFUSED_RATE_DOWN_PAYMENT_MAX"]>0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0), -1.0 ), ((data["REFUSED_AMT_DOWN_PAYMENT_MAX"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) )))*1.))))) 
    v["i305"] = 0.099795*np.tanh(np.where((((data["AMT_ANNUITY"]) + (data["CC_NAME_CONTRACT_STATUS_Refused_MAX"]))/2.0)<0, ((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + ((((1.0) + (data["NAME_INCOME_TYPE_Maternity_leave"]))/2.0)))/2.0)) < (data["CC_NAME_CONTRACT_STATUS_Refused_MAX"]))*1.), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"])) )) 
    v["i306"] = 0.099949*np.tanh(np.where((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > ((-1.0*((np.maximum(((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])), (((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > (((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) * 2.0)))*1.)))))))))*1.)>0, -2.0, (((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) > (data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]))*1.) )) 
    v["i307"] = 0.082296*np.tanh(((np.minimum((((((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > (data["CC_NAME_CONTRACT_STATUS_Signed_VAR"]))*1.)) * (data["CC_NAME_CONTRACT_STATUS_Completed_VAR"])))), ((((((data["CC_NAME_CONTRACT_STATUS_Completed_VAR"]) - (((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0)) * 2.0)))) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))))) * 2.0)) 
    v["i308"] = 0.086999*np.tanh(np.maximum(((data["CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN"])), (((((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) > (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]<0, data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"], ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) * ((((2.0) > (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))*1.))) )))*1.)) * 2.0)) * 2.0))))) 
    v["i309"] = 0.099670*np.tanh((-1.0*((np.maximum(((np.where(data["CC_NAME_CONTRACT_STATUS_Signed_VAR"]<0, np.where(data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"] < -99998, (0.0), 0.318310 ), np.maximum(((data["CC_NAME_CONTRACT_STATUS_Signed_VAR"])), (((1.0)))) ))), ((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]))))))) 
    v["i310"] = 0.095000*np.tanh((((((((((2.0) - (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))) < (np.where(np.where(data["CC_AMT_DRAWINGS_OTHER_CURRENT_MAX"]>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], 2.0 )>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], 3.0 )))*1.)) * 2.0)) * 2.0)) 
    v["i311"] = 0.098999*np.tanh(((3.0) * (np.minimum(((((((3.0) * (((data["NAME_INCOME_TYPE_Student"]) * (data["NEW_DOC_IND_KURT"]))))) * 2.0))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["NAME_CONTRACT_TYPE_Revolving_loans"]))*1.))))))))) 
    v["i312"] = 0.098189*np.tanh(((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, (((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0), (((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (8.0), (((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0) )) < (data["CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN"]))*1.) )) * 2.0)) 
    v["i313"] = 0.099849*np.tanh((-1.0*(((((data["CC_NAME_CONTRACT_STATUS_Completed_VAR"]) > (np.where(data["CC_NAME_CONTRACT_STATUS_Active_VAR"]<0, (-1.0*(((((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0)) > (data["CC_NAME_CONTRACT_STATUS_Completed_VAR"]))*1.)))), ((0.636620) + (2.0)) )))*1.))))) 
    v["i314"] = 0.099020*np.tanh(np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"] < -99998, 0.0, ((((((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) - (data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]))) - (data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]))) - (((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) / 2.0))) )) 
    v["i315"] = 0.098960*np.tanh(np.minimum(((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]))) - ((((3.141593) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))), (((((3.0) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.))))) 
    v["i316"] = 0.066799*np.tanh((((((np.maximum(((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"])), ((1.570796)))) > (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]>0, np.maximum(((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"])), ((np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]<0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"], data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"] )))), 2.0 )))*1.)) / 2.0)) 
    v["i317"] = 0.099501*np.tanh((-1.0*(((((((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (4.0), (-1.0*(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))*1.)))) )))*1.)) * ((6.73373746871948242))))))) 
    v["i318"] = 0.098100*np.tanh(((((((np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]<0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], 0.318310 )) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))/2.0)) > ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) > (np.tanh((np.maximum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"])))))))*1.)))*1.)) 
    v["i319"] = 0.095460*np.tanh((((((3.0)) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * ((((((11.21043491363525391)) + ((11.21043491363525391)))) * ((-1.0*(((((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (3.141593))*1.))))))))) 
    v["i320"] = 0.089963*np.tanh(((np.minimum(((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) * (((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))))) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))), (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0))*1.))))) * 2.0)) 
    v["i321"] = 0.084251*np.tanh(((((data["ORGANIZATION_TYPE_Industry__type_12"]) + (((((np.maximum((((((3.141593) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.))), ((data["ORGANIZATION_TYPE_Industry__type_11"])))) * 2.0)) * 2.0)))) * ((-1.0*((data["AMT_ANNUITY"])))))) 
    v["i322"] = 0.070000*np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((np.minimum((((((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((((3.141593) / 2.0))))) < (data["AMT_ANNUITY"]))*1.))), ((((0.318310) * (data["AMT_ANNUITY"])))))) - (-2.0))))*1.)) 
    v["i323"] = 0.060300*np.tanh(((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) - (((-2.0) * (np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, data["POS_SK_DPD_MEAN"], (2.30259108543395996) )))))) > ((6.0)))*1.)))) * 2.0)) * 2.0)) 
    v["i324"] = 0.060646*np.tanh(np.where(((((data["POS_SK_DPD_MEAN"]) - (np.tanh((data["POS_SK_DPD_MAX"]))))) * (data["POS_SK_DPD_MEAN"]))<0, ((data["POS_SK_DPD_MAX"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["POS_SK_DPD_MAX"])))), data["NEW_ANNUITY_TO_INCOME_RATIO"] )) 
    v["i325"] = 0.098599*np.tanh((((((data["DAYS_BIRTH"]) > (np.maximum(((((data["DAYS_BIRTH"]) * 2.0))), ((data["ORGANIZATION_TYPE_University"])))))*1.)) - (((((((np.maximum(((((data["ORGANIZATION_TYPE_Industry__type_12"]) / 2.0))), ((data["ORGANIZATION_TYPE_University"])))) / 2.0)) / 2.0)) / 2.0)))) 
    v["i326"] = 0.070002*np.tanh(np.where((((data["ORGANIZATION_TYPE_Industry__type_1"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)>0, data["ORGANIZATION_TYPE_Industry__type_1"], np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["NAME_INCOME_TYPE_Working"], ((((((np.tanh((1.0))) + (data["NAME_INCOME_TYPE_Working"]))/2.0)) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.) ) )) 
    v["i327"] = 0.091899*np.tanh(((((((((((((((8.0)) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)) + (data["ORGANIZATION_TYPE_Industry__type_3"]))/2.0)) * ((-1.0*((data["NEW_CREDIT_TO_INCOME_RATIO"])))))) + (data["ORGANIZATION_TYPE_Industry__type_3"]))/2.0)) * ((-1.0*((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) 
    v["i328"] = 0.097300*np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, np.maximum(((data["ORGANIZATION_TYPE_Legal_Services"])), ((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) * (data["NAME_INCOME_TYPE_Working"]))))), (-1.0*(((((((((data["NAME_INCOME_TYPE_Working"]) * (data["OCCUPATION_TYPE_Low_skill_Laborers"]))) * 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))) )) 
    v["i329"] = 0.099970*np.tanh(((((((10.0)) < (np.maximum(((((data["AMT_ANNUITY"]) * (data["NEW_CREDIT_TO_INCOME_RATIO"])))), (((((((9.67386054992675781)) - (((data["AMT_ANNUITY"]) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))))) - (1.0)))))))*1.)) * 2.0)) 
    v["i330"] = 0.100000*np.tanh(np.minimum(((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (0.636620))*1.)) > (((1.570796) + ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))*1.))), ((((1.570796) + ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))) 
    v["i331"] = 0.099985*np.tanh(((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (np.where(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["ORGANIZATION_TYPE_Trade__type_4"], data["AMT_INCOME_TOTAL"] )>0, (-1.0*((((data["AMT_INCOME_TOTAL"]) + (data["NAME_CONTRACT_TYPE_Revolving_loans"]))))), ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (data["ORGANIZATION_TYPE_Trade__type_4"])) )))) 
    v["i332"] = 0.099999*np.tanh(((((((data["NEW_INC_PER_CHLD"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["OCCUPATION_TYPE_Low_skill_Laborers"], data["EXT_SOURCE_1"] )))*1.)))) * (2.0))) * ((((-1.0) < (data["EXT_SOURCE_1"]))*1.)))) 
    v["i333"] = 0.099500*np.tanh(((((((6.92336940765380859)) < (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0)))*1.)) - (((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + ((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))/2.0)))/2.0)) > (1.0))*1.)))) 
    v["i334"] = 0.064600*np.tanh((-1.0*(((((((data["AMT_CREDIT"]) - ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (np.minimum(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["CNT_CHILDREN"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] ))), ((data["AMT_ANNUITY"])))))*1.)))) > (np.tanh((1.570796))))*1.))))) 
    v["i335"] = 0.092697*np.tanh(((((np.where(data["NAME_FAMILY_STATUS_Married"]>0, (((data["NAME_INCOME_TYPE_Student"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.), np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]>0, data["NEW_CREDIT_TO_INCOME_RATIO"], np.where(data["AMT_ANNUITY"]>0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NAME_FAMILY_STATUS_Married"] ) ) )) / 2.0)) / 2.0)) 
    v["i336"] = 0.090054*np.tanh(np.where(data["AMT_ANNUITY"]<0, np.where(np.where((((data["AMT_ANNUITY"]) < (data["NAME_FAMILY_STATUS_Single___not_married"]))*1.)>0, data["NAME_FAMILY_STATUS_Single___not_married"], ((data["CNT_CHILDREN"]) / 2.0) )<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["CNT_CHILDREN"] ), ((data["NAME_FAMILY_STATUS_Single___not_married"]) / 2.0) )) 
    v["i337"] = 0.085501*np.tanh(((np.minimum(((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) / 2.0))), ((((np.where((((data["FLAG_EMP_PHONE"]) < (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))*1.)>0, data["CODE_GENDER"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) * (data["CODE_GENDER"])))))) / 2.0)) 
    v["i338"] = 0.010999*np.tanh(np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"]>0, np.where(data["ORGANIZATION_TYPE_Trade__type_4"]>0, data["ORGANIZATION_TYPE_Trade__type_4"], (((-1.0*((data["ORGANIZATION_TYPE_Kindergarten"])))) + ((-1.0*(((((data["ORGANIZATION_TYPE_Kindergarten"]) < (data["AMT_ANNUITY"]))*1.)))))) ), (-1.0*((data["ORGANIZATION_TYPE_Trade__type_4"]))) )) 
    v["i339"] = 0.096004*np.tanh(((((np.where(((data["NAME_INCOME_TYPE_Maternity_leave"]) * (data["WEEKDAY_APPR_PROCESS_START_MONDAY"]))>0, np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), (((-1.0*((data["NAME_INCOME_TYPE_Maternity_leave"])))))), ((data["NAME_INCOME_TYPE_Student"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])) )) * (data["ORGANIZATION_TYPE_Kindergarten"]))) / 2.0)) 
    v["i340"] = 0.095596*np.tanh(np.where((((data["AMT_GOODS_PRICE"]) > ((0.41270980238914490)))*1.)>0, data["NEW_CREDIT_TO_GOODS_RATIO"], np.where(data["AMT_GOODS_PRICE"]>0, data["AMT_GOODS_PRICE"], ((np.maximum(((np.tanh((1.0)))), ((data["NEW_CREDIT_TO_GOODS_RATIO"])))) - (1.0)) ) )) 
    v["i341"] = 0.089984*np.tanh(((0.636620) * ((-1.0*((np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"]>0, data["AMT_ANNUITY"], ((data["AMT_ANNUITY"]) * (np.where(data["NEW_DOC_IND_KURT"]>0, (((1.0) < (data["NAME_CONTRACT_TYPE_Revolving_loans"]))*1.), -2.0 ))) ))))))) 
    v["i342"] = 0.074030*np.tanh((((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, np.where(data["NAME_INCOME_TYPE_Pensioner"]<0, 1.0, data["NEW_CREDIT_TO_ANNUITY_RATIO"] ), data["AMT_ANNUITY"] )) < (-1.0))*1.)) * (((-1.0) - (data["DAYS_BIRTH"]))))) * 2.0)) 
    v["i343"] = 0.089964*np.tanh(((data["ORGANIZATION_TYPE_Trade__type_2"]) * (((((1.0)) + (np.where(data["ORGANIZATION_TYPE_Trade__type_2"]<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.minimum(((((data["FLAG_PHONE"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((((data["ORGANIZATION_TYPE_Trade__type_2"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) )))/2.0)))) 
    v["i344"] = 0.000099*np.tanh((-1.0*((((np.where(data["FLAG_PHONE"]<0, data["ORGANIZATION_TYPE_Trade__type_2"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) * (np.where((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((data["FLAG_PHONE"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0) ))))))) 
    v["i345"] = 0.095300*np.tanh(np.where(data["FLAG_PHONE"]>0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((-1.0*((-2.0)))))*1.), (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((data["ORGANIZATION_TYPE_Trade__type_2"]) * 2.0)))*1.) ), data["ORGANIZATION_TYPE_Trade__type_2"] )) 
    v["i346"] = 0.003000*np.tanh(((((np.where(((data["NEW_DOC_IND_KURT"]) + (((data["NEW_DOC_IND_KURT"]) * (data["AMT_ANNUITY"]))))>0, data["NEW_DOC_IND_KURT"], np.where(data["AMT_ANNUITY"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["AMT_ANNUITY"] ) )) / 2.0)) / 2.0)) 
    v["i347"] = 0.089001*np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((((data["AMT_ANNUITY"]) * (data["AMT_ANNUITY"]))) * ((((2.0) < (((data["AMT_ANNUITY"]) * ((-1.0*((((data["NEW_DOC_IND_KURT"]) + (data["NAME_CONTRACT_TYPE_Cash_loans"])))))))))*1.)))))) 
    v["i348"] = 0.072951*np.tanh(((np.where(data["FLAG_DOCUMENT_3"]<0, (((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) + (data["NEW_DOC_IND_KURT"]))) / 2.0)) < (data["FLAG_DOCUMENT_3"]))*1.), ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) + (data["NEW_DOC_IND_KURT"])) )) * 2.0)) 
    v["i349"] = 0.098823*np.tanh(np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0*(((((data["NAME_INCOME_TYPE_Student"]) > ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (np.maximum(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((np.tanh((1.570796)))))))/2.0)))*1.)))) )) 
    v["i350"] = 0.020000*np.tanh((((((((((np.where(((data["NAME_INCOME_TYPE_Maternity_leave"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))<0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["NAME_INCOME_TYPE_Student"] )) > (data["NAME_INCOME_TYPE_Student"]))*1.)) * 2.0)) * 2.0)) - (data["NAME_INCOME_TYPE_Student"]))) 
    v["i351"] = 0.098004*np.tanh(np.where(data["AMT_ANNUITY"]>0, ((np.where(((((data["ORGANIZATION_TYPE_Trade__type_3"]) / 2.0)) + (data["AMT_ANNUITY"]))<0, (14.66671943664550781), data["ORGANIZATION_TYPE_Mobile"] )) * (-1.0)), data["ORGANIZATION_TYPE_Mobile"] )) 
    v["i352"] = 0.099584*np.tanh(np.where(data["ORGANIZATION_TYPE_Trade__type_3"]<0, (((((0.636620) < (data["LIVINGAPARTMENTS_AVG"]))*1.)) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["LIVINGAPARTMENTS_AVG"], -1.0 ))), ((data["LIVINGAPARTMENTS_AVG"]) - (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))) )) 
    v["i353"] = 0.000020*np.tanh(((((((((((data["LIVINGAPARTMENTS_MEDI"]) > (2.0))*1.)) < ((((-2.0) < (data["LIVINGAPARTMENTS_MEDI"]))*1.)))*1.)) * (((data["NONLIVINGAPARTMENTS_AVG"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) * 2.0)) 
    v["i354"] = 0.065970*np.tanh(((data["NONLIVINGAPARTMENTS_AVG"]) * (((np.where(data["NONLIVINGAPARTMENTS_MODE"]<0, data["AMT_ANNUITY"], (((np.maximum(((data["NONLIVINGAPARTMENTS_MODE"])), ((data["LIVINGAPARTMENTS_MODE"])))) > (2.0))*1.) )) * ((((data["LIVINGAPARTMENTS_MODE"]) > (-1.0))*1.)))))) 
    v["i355"] = 0.086500*np.tanh(((((np.minimum((((((-1.0) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.))), ((((np.maximum(((data["LIVINGAPARTMENTS_AVG"])), (((((((0.636620) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < (data["AMT_ANNUITY"]))*1.))))) * 2.0))))) * 2.0)) * 2.0)) 
    v["i356"] = 0.098650*np.tanh(np.minimum(((np.where((((1.570796) + (data["AMT_ANNUITY"]))/2.0)<0, (((3.0) + (data["NONLIVINGAPARTMENTS_MEDI"]))/2.0), (((data["LIVINGAPARTMENTS_MODE"]) > ((3.0)))*1.) ))), (((((data["NONLIVINGAPARTMENTS_MEDI"]) > (data["AMT_ANNUITY"]))*1.))))) 
    v["i357"] = 0.018997*np.tanh(np.minimum((((-1.0*((((data["NONLIVINGAPARTMENTS_AVG"]) * ((((data["LIVINGAPARTMENTS_MEDI"]) > (0.636620))*1.)))))))), ((np.where(((data["NONLIVINGAPARTMENTS_AVG"]) + (((data["LIVINGAPARTMENTS_MODE"]) / 2.0))) < -99998, 3.0, data["NONLIVINGAPARTMENTS_AVG"] ))))) 
    v["i358"] = 0.042000*np.tanh((-1.0*((((((((data["LIVINGAPARTMENTS_MEDI"]) > (data["AMT_ANNUITY"]))*1.)) < (np.where(data["NONLIVINGAPARTMENTS_AVG"]<0, (((data["LIVINGAPARTMENTS_MEDI"]) + ((0.22441512346267700)))/2.0), (-1.0*((((data["AMT_ANNUITY"]) / 2.0)))) )))*1.))))) 
    v["i359"] = 0.097998*np.tanh(np.where(data["INSTAL_PAYMENT_DIFF_VAR"] < -99998, (((-1.0*((data["AMT_ANNUITY"])))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["INSTAL_PAYMENT_DIFF_VAR"]) > (((((data["INSTAL_PAYMENT_DIFF_VAR"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) / 2.0)))*1.))) )) 
    v["i360"] = 0.070050*np.tanh(np.where(((((data["INSTAL_PAYMENT_DIFF_VAR"]) - (((data["INSTAL_DBD_STD"]) + (0.318310))))) + (data["INSTAL_PAYMENT_DIFF_VAR"]))>0, np.where(data["INSTAL_DBD_STD"]<0, data["INSTAL_DBD_STD"], (10.57927513122558594) ), 0.318310 )) 
    v["i361"] = 0.096000*np.tanh(np.minimum(((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))), ((((((np.where(data["CC_NAME_CONTRACT_STATUS_Active_VAR"]>0, data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"], (14.66551780700683594) )) - (data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]))) * ((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > (2.0))*1.))))))) 
    v["i362"] = 0.099265*np.tanh(((((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["ORGANIZATION_TYPE_Industry__type_1"])))) * (((data["ORGANIZATION_TYPE_Industry__type_1"]) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (11.86987972259521484), np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) )))))) * 2.0)) 
    v["i363"] = 0.095700*np.tanh(np.where(data["AMT_ANNUITY"]<0, np.where(np.maximum(((data["NEW_LIVE_IND_SUM"])), ((data["NAME_INCOME_TYPE_Maternity_leave"])))<0, np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["FLAG_EMAIL"] ), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0) ), (-1.0*((data["ORGANIZATION_TYPE_Trade__type_4"]))) )) 
    v["i364"] = 0.096810*np.tanh(((np.where((-1.0*((data["AMT_ANNUITY"])))<0, (-1.0*(((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) > (1.570796))*1.)))), (((((((2.0)) < (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))*1.)) < (data["INSTAL_PAYMENT_PERC_STD"]))*1.) )) * (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))) 
    v["i365"] = 0.100000*np.tanh(((((1.0) - (np.maximum(((np.minimum(((data["FONDKAPREMONT_MODE_reg_oper_spec_account"])), ((data["LIVINGAPARTMENTS_MODE"]))))), (((((np.where(data["AMT_ANNUITY"]<0, data["LIVINGAPARTMENTS_MODE"], ((data["LIVINGAPARTMENTS_MODE"]) / 2.0) )) < ((3.0)))*1.))))))) * 2.0)) 
    v["i366"] = 0.018860*np.tanh((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((((np.maximum((((((10.0)) + (data["LIVINGAPARTMENTS_AVG"])))), ((np.tanh((np.where(data["AMT_ANNUITY"]>0, np.tanh(((10.0))), data["AMT_ANNUITY"] ))))))) + ((4.0)))/2.0)))*1.)) 
    v["i367"] = 0.099050*np.tanh((-1.0*((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["NAME_EDUCATION_TYPE_Academic_degree"], ((((((((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0)) - (data["NAME_INCOME_TYPE_Student"]))) < ((((-2.0) + (data["INSTAL_PAYMENT_DIFF_VAR"]))/2.0)))*1.) ))))) 
    v["i368"] = 0.095009*np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["FONDKAPREMONT_MODE_org_spec_account"], (((data["OCCUPATION_TYPE_Cleaning_staff"]) < (((data["OCCUPATION_TYPE_Cleaning_staff"]) * ((((3.141593) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)))))*1.) )) * (((data["OCCUPATION_TYPE_Cleaning_staff"]) - (data["AMT_ANNUITY"]))))) 
    v["i369"] = 0.082589*np.tanh((-1.0*((((3.141593) * ((((3.141593) < (np.where(data["OCCUPATION_TYPE_Cleaning_staff"]>0, ((3.141593) * ((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + (3.0))/2.0))), data["INSTAL_PAYMENT_DIFF_VAR"] )))*1.))))))) 
    v["i370"] = 0.081997*np.tanh(np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]>0, data["ORGANIZATION_TYPE_Transport__type_3"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((data["ORGANIZATION_TYPE_Transport__type_3"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))))) )) 
    v["i371"] = 0.091909*np.tanh(np.where((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (data["AMT_ANNUITY"]))*1.)>0, data["NAME_INCOME_TYPE_Maternity_leave"], ((((8.14943599700927734)) < (((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (((np.tanh((data["AMT_ANNUITY"]))) - ((8.27048969268798828)))))))*1.) )) 
    v["i372"] = 0.090395*np.tanh(((data["NAME_TYPE_SUITE_Children"]) * (((data["NAME_TYPE_SUITE_Children"]) * ((((np.tanh((data["AMT_ANNUITY"]))) + (((((1.47119557857513428)) + (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NEW_ANNUITY_TO_INCOME_RATIO"] )))/2.0)))/2.0)))))) 
    v["i373"] = 0.092002*np.tanh(np.minimum((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (2.0))*1.))), (((((((1.0) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) - (np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NAME_TYPE_SUITE_Spouse__partner"]))*1.)>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["ORGANIZATION_TYPE_Trade__type_4"] ))))))) 
    v["i374"] = 0.089650*np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (2.0))*1.)) * ((-1.0*((((((((((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]>0, ((data["AMT_ANNUITY"]) * 2.0), data["NAME_TYPE_SUITE_Spouse__partner"] )) * 2.0)) * 2.0)) * 2.0)) * 2.0))))))) 
    v["i375"] = 0.099860*np.tanh(((((np.where(data["NAME_TYPE_SUITE_Spouse__partner"]>0, 2.0, data["NEW_CREDIT_TO_INCOME_RATIO"] )) - (data["NEW_CREDIT_TO_INCOME_RATIO"]))) * ((((((((data["ORGANIZATION_TYPE_Government"]) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) - (data["ORGANIZATION_TYPE_Government"]))))) 
    v["i376"] = 0.028342*np.tanh((-1.0*((((((np.where(np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, data["NAME_TYPE_SUITE_Spouse__partner"], np.minimum(((data["FLAG_EMAIL"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))) )<0, 0.0, data["FLAG_EMAIL"] )) * 2.0)) * 2.0))))) 
    v["i377"] = 0.049000*np.tanh((((-1.0*(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((((((2.0) > (data["AMT_CREDIT"]))*1.)) + ((((np.minimum(((3.0)), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) < (data["AMT_CREDIT"]))*1.)))))*1.))))) / 2.0)) 
    v["i378"] = 0.097398*np.tanh((((((np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + ((((((data["NAME_TYPE_SUITE_Other_B"]) * (data["NAME_TYPE_SUITE_Other_B"]))) < (data["EXT_SOURCE_1"]))*1.)))/2.0)<0, data["NAME_INCOME_TYPE_Student"], 1.570796 )) < (data["EXT_SOURCE_1"]))*1.)) * (-2.0))) 
    v["i379"] = 0.099776*np.tanh((((-1.0*((data["OCCUPATION_TYPE_Laborers"])))) * ((((((((((np.where(data["EXT_SOURCE_1"] < -99998, (1.77269029617309570), (((data["OCCUPATION_TYPE_Laborers"]) < (2.0))*1.) )) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * 2.0)) * 2.0)) * 2.0)))) 
    v["i380"] = 0.087500*np.tanh((-1.0*(((((data["NAME_INCOME_TYPE_Student"]) + ((((np.where(data["AMT_ANNUITY"]<0, data["NAME_INCOME_TYPE_Student"], np.maximum(((np.maximum(((data["AMT_ANNUITY"])), ((data["NAME_INCOME_TYPE_Student"]))))), ((data["OCCUPATION_TYPE_Laborers"]))) )) < (data["AMT_CREDIT"]))*1.)))/2.0))))) 
    v["i381"] = 0.061010*np.tanh((-1.0*((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["ORGANIZATION_TYPE_Trade__type_7"]))) * (np.where(data["AMT_ANNUITY"]>0, np.where(data["ORGANIZATION_TYPE_Trade__type_7"]>0, data["DAYS_BIRTH"], data["NAME_TYPE_SUITE_Other_B"] ), data["ORGANIZATION_TYPE_Trade__type_7"] ))))))) 
    v["i382"] = 0.094702*np.tanh(np.where(data["DAYS_BIRTH"]>0, np.where(data["OCCUPATION_TYPE_Drivers"]>0, data["ORGANIZATION_TYPE_Trade__type_7"], (-1.0*((data["ORGANIZATION_TYPE_Trade__type_7"]))) ), np.tanh((np.tanh(((((data["DAYS_BIRTH"]) < ((-1.0*((1.570796)))))*1.))))) )) 
    v["i383"] = 0.099500*np.tanh(np.where((((data["AMT_ANNUITY"]) < ((1.65035283565521240)))*1.)>0, np.where(data["ORGANIZATION_TYPE_Trade__type_7"]>0, (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["AMT_ANNUITY"]))*1.), ((((1.55915415287017822)) < (data["AMT_ANNUITY"]))*1.) ), data["FLAG_DOCUMENT_3"] )) 
    v["i384"] = 0.072983*np.tanh((((12.58773136138916016)) * (np.where(data["DAYS_REGISTRATION"]>0, data["NAME_INCOME_TYPE_Maternity_leave"], (((((((((7.83792781829833984)) + (-2.0))/2.0)) < (data["AMT_ANNUITY"]))*1.)) * 2.0) )))) 
    v["i385"] = 0.062950*np.tanh(((((((((np.minimum(((-1.0)), ((((data["AMT_ANNUITY"]) * 2.0))))) - (data["DAYS_ID_PUBLISH"]))) < (((((((0.318310) < (data["AMT_ANNUITY"]))*1.)) < (data["AMT_ANNUITY"]))*1.)))*1.)) + (-1.0))/2.0)) 
    v["i386"] = 0.090988*np.tanh(((np.minimum(((np.where(data["LIVINGAPARTMENTS_MEDI"] < -99998, (((data["DAYS_BIRTH"]) < (-1.0))*1.), np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["DAYS_BIRTH"], data["LIVINGAPARTMENTS_MEDI"] ) ))), (((((data["DAYS_BIRTH"]) < (0.636620))*1.))))) / 2.0)) 
    v["i387"] = 0.099989*np.tanh(np.minimum((((((data["FLOORSMIN_MEDI"]) < ((((np.where(data["LIVINGAREA_MODE"]>0, data["ELEVATORS_MODE"], data["LIVINGAREA_MODE"] )) > (data["ELEVATORS_MEDI"]))*1.)))*1.))), (((((np.minimum(((data["FLOORSMIN_MEDI"])), ((data["ELEVATORS_MEDI"])))) > (data["ENTRANCES_MODE"]))*1.))))) 
    v["i388"] = 0.099710*np.tanh(np.maximum(((np.tanh((data["NEW_CAR_TO_EMPLOY_RATIO"])))), ((np.minimum((((((((data["NAME_INCOME_TYPE_Pensioner"]) + (data["EXT_SOURCE_1"]))/2.0)) * (((((data["NAME_INCOME_TYPE_Pensioner"]) * 2.0)) * 2.0))))), ((np.maximum(((0.0)), ((data["NAME_INCOME_TYPE_Pensioner"])))))))))) 
    v["i389"] = 0.080000*np.tanh(np.minimum((((((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) < (data["DAYS_EMPLOYED"]))*1.))), ((((np.where(((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) - (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))<0, data["NAME_INCOME_TYPE_Pensioner"], 3.141593 )) - (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) - (data["NAME_INCOME_TYPE_Pensioner"])))))))) 
    v["i390"] = 0.100000*np.tanh((-1.0*((((np.where(((((((1.0) + (1.570796))/2.0)) + (data["NEW_SOURCES_PROD"]))/2.0)>0, data["DAYS_BIRTH"], (((-1.0*((0.318310)))) * (data["DAYS_BIRTH"])) )) * 2.0))))) 
    v["i391"] = 0.084999*np.tanh(((((-1.0*((data["DAYS_BIRTH"])))) + (((((((data["DAYS_EMPLOYED"]) > (0.636620))*1.)) > (((((-1.0*((data["DAYS_BIRTH"])))) + ((((0.636620) > (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))*1.)))/2.0)))*1.)))/2.0)) 
    v["i392"] = 0.090389*np.tanh(np.where(data["NEW_SOURCES_PROD"]>0, ((data["EXT_SOURCE_3"]) - (data["NEW_SOURCES_PROD"])), (((data["NEW_SOURCES_PROD"]) > (np.tanh((((((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) - (1.0))) - (data["NEW_SOURCES_PROD"]))))))*1.) )) 
    v["i393"] = 0.099520*np.tanh((((np.minimum(((((((2.0)) > (data["CLOSED_AMT_CREDIT_SUM_MEAN"]))*1.))), ((np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]<0, data["CLOSED_AMT_CREDIT_SUM_MEAN"], data["CLOSED_AMT_CREDIT_SUM_SUM"] ))))) > ((((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) > (np.tanh((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))))*1.)) / 2.0)))*1.)) 
    v["i394"] = 0.075001*np.tanh(((((((((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["CODE_GENDER"]))*1.)) > (data["CODE_GENDER"]))*1.)) * (((np.where(data["NEW_CAR_TO_BIRTH_RATIO"] < -99998, 0.0, data["CODE_GENDER"] )) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))) / 2.0)) 
    v["i395"] = 0.099623*np.tanh(np.where((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (3.141593))*1.)>0, (-1.0*((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))), ((np.tanh((np.tanh((np.tanh((((data["EXT_SOURCE_3"]) * ((-1.0*((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))))))))))) / 2.0) )) 
    v["i396"] = 0.099000*np.tanh(np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, -2.0, ((np.where(data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]<0, (((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) < (data["AMT_REQ_CREDIT_BUREAU_DAY"]))*1.), ((data["AMT_REQ_CREDIT_BUREAU_DAY"]) - ((((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) < (data["AMT_REQ_CREDIT_BUREAU_YEAR"]))*1.))) )) / 2.0) )) 
    v["i397"] = 0.099702*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_MON"]>0, ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) * (((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) + (data["AMT_REQ_CREDIT_BUREAU_MON"])))), (-1.0*((np.maximum((((((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) < (data["AMT_REQ_CREDIT_BUREAU_MON"]))*1.))), ((((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) * 2.0))))))) )) 
    v["i398"] = 0.095995*np.tanh(np.minimum(((np.where(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) - (data["AMT_REQ_CREDIT_BUREAU_MON"]))>0, (-1.0*((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))), (((6.0)) - (data["AMT_REQ_CREDIT_BUREAU_MON"])) ))), (((((1.570796) < (data["AMT_REQ_CREDIT_BUREAU_QRT"]))*1.))))) 
    v["i399"] = 0.086499*np.tanh((((((-1.0*((np.where((((0.318310) < (((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) - (3.141593))))*1.)>0, data["BURO_CREDIT_TYPE_Car_loan_MEAN"], (-1.0*(((((2.0) < (data["BURO_CREDIT_TYPE_Car_loan_MEAN"]))*1.)))) ))))) * 2.0)) * 2.0)) 
    v["i400"] = 0.047958*np.tanh(np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"]<0, np.minimum((((((((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) * (data["CNT_CHILDREN"]))) < (data["AMT_REQ_CREDIT_BUREAU_WEEK"]))*1.))), ((((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) * (data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]))))), ((data["CNT_CHILDREN"]) + (-2.0)) )) 
    v["i401"] = 0.089300*np.tanh((((((data["EXT_SOURCE_1"]) > (np.where(data["EXT_SOURCE_1"]>0, (((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) < (data["AMT_ANNUITY"]))*1.), data["AMT_ANNUITY"] )))*1.)) * ((((((data["NAME_INCOME_TYPE_Student"]) > (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)) + (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))) 
    v["i402"] = 0.098002*np.tanh(((np.where((-1.0*((data["AMT_REQ_CREDIT_BUREAU_WEEK"])))>0, (((3.141593) < (np.where(data["AMT_REQ_CREDIT_BUREAU_YEAR"]<0, data["AMT_REQ_CREDIT_BUREAU_QRT"], data["AMT_REQ_CREDIT_BUREAU_DAY"] )))*1.), data["AMT_REQ_CREDIT_BUREAU_QRT"] )) * 2.0)) 
    v["i403"] = 0.056801*np.tanh((-1.0*(((((np.maximum(((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"])), ((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))) > (np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]<0, (13.24810218811035156), 0.318310 )))*1.))))) 
    v["i404"] = 0.091995*np.tanh(((((((((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) + ((0.24336224794387817)))) > (data["AMT_REQ_CREDIT_BUREAU_YEAR"]))*1.)) < (((((((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) > (data["AMT_REQ_CREDIT_BUREAU_HOUR"]))*1.)) > ((((2.0) < (data["AMT_REQ_CREDIT_BUREAU_QRT"]))*1.)))*1.)))*1.)) 
    v["i405"] = 0.013991*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_WEEK"] < -99998, 0.318310, ((np.maximum(((((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) - (data["AMT_REQ_CREDIT_BUREAU_DAY"])))), ((data["AMT_REQ_CREDIT_BUREAU_MON"])))) * (((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) - (data["AMT_REQ_CREDIT_BUREAU_DAY"])))) )) 
    v["i406"] = 0.018490*np.tanh((((-1.0*(((3.0))))) * (((((((3.0)) < ((((((np.maximum(((data["AMT_REQ_CREDIT_BUREAU_YEAR"])), (((1.03554749488830566))))) * 2.0)) + ((((data["AMT_REQ_CREDIT_BUREAU_MON"]) + (data["BURO_CNT_CREDIT_PROLONG_SUM"]))/2.0)))/2.0)))*1.)) * 2.0)))) 
    v["i407"] = 0.080770*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]>0, np.minimum(((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((data["AMT_REQ_CREDIT_BUREAU_MON"]))))), ((data["AMT_REQ_CREDIT_BUREAU_YEAR"]))), (((2.67786097526550293)) * ((-1.0*(((((data["AMT_REQ_CREDIT_BUREAU_HOUR"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.)))))) )) 
    v["i408"] = 0.086000*np.tanh((((((6.0)) + (data["AMT_REQ_CREDIT_BUREAU_DAY"]))) * ((-1.0*(((((np.where(data["EXT_SOURCE_1"] < -99998, ((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) - ((-1.0*((data["AMT_REQ_CREDIT_BUREAU_MON"]))))), data["EXT_SOURCE_1"] )) > ((1.48802196979522705)))*1.))))))) 
    v["i409"] = 0.092999*np.tanh(np.maximum((((-1.0*((((1.0) / 2.0)))))), ((np.where((((data["EXT_SOURCE_1"]) + (2.0))/2.0)>0, (-1.0*((((data["EXT_SOURCE_1"]) - ((-1.0*((data["DAYS_BIRTH"])))))))), data["DAYS_BIRTH"] ))))) 
    v["i410"] = 0.099581*np.tanh(np.where(((data["ORGANIZATION_TYPE_XNA"]) * (-1.0))>0, ((np.where(data["DAYS_BIRTH"]>0, 0.318310, data["DAYS_BIRTH"] )) * (data["CODE_GENDER"])), (((((-1.0*((data["DAYS_BIRTH"])))) / 2.0)) / 2.0) )) 
    v["i411"] = 0.097800*np.tanh(np.where(((data["NEW_SOURCES_PROD"]) + (data["EXT_SOURCE_1"]))<0, np.where(data["EXT_SOURCE_1"] < -99998, 0.0, (((data["DAYS_BIRTH"]) < (np.tanh(((((data["EXT_SOURCE_1"]) < (data["NAME_INCOME_TYPE_Student"]))*1.)))))*1.) ), data["AMT_ANNUITY"] )) 
    v["i412"] = 0.030003*np.tanh(np.where((((data["LIVINGAREA_MODE"]) + (np.where(data["LIVINGAREA_MEDI"]>0, -2.0, data["LIVINGAREA_AVG"] )))/2.0)>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where(data["LIVINGAREA_MODE"]>0, data["NAME_INCOME_TYPE_Student"], (-1.0*((((0.318310) / 2.0)))) ) )) 
    v["i413"] = 0.089859*np.tanh((((-1.0*((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["CODE_GENDER"]))>0, 0.636620, data["DAYS_BIRTH"] ), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["CODE_GENDER"])) ))))) / 2.0)) 
    v["i414"] = 0.099998*np.tanh(((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]>0, 0.318310, (-1.0*(((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) / 2.0)) < (data["NEW_CAR_TO_EMPLOY_RATIO"]))*1.)))) )) - (np.where((-1.0*((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"])))<0, data["AMT_ANNUITY"], data["NAME_INCOME_TYPE_Student"] )))) 
    v["i415"] = 0.100000*np.tanh(((data["WALLSMATERIAL_MODE_Stone__brick"]) * (np.where((((2.0) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)>0, (((data["LIVINGAREA_MODE"]) > (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) + (2.0))))*1.), 3.141593 )))) 
    v["i416"] = 0.010500*np.tanh(((((data["AMT_ANNUITY"]) * ((((13.51322269439697266)) * (np.where(np.where(data["LIVINGAREA_MEDI"] < -99998, data["AMT_ANNUITY"], data["NEW_ANNUITY_TO_INCOME_RATIO"] )>0, data["NAME_INCOME_TYPE_Student"], (((data["LIVINGAREA_MEDI"]) > ((7.19499874114990234)))*1.) )))))) * 2.0)) 
    v["i417"] = 0.097300*np.tanh(((np.where(np.where(data["FONDKAPREMONT_MODE_reg_oper_account"]<0, (7.0), data["AMT_ANNUITY"] )<0, (((data["LIVINGAREA_MEDI"]) > (data["HOUSETYPE_MODE_block_of_flats"]))*1.), ((((7.0)) < (data["LIVINGAREA_MEDI"]))*1.) )) * ((7.0)))) 
    v["i418"] = 0.050100*np.tanh(((((data["ACTIVE_AMT_ANNUITY_MAX"]) + (np.minimum(((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) + (data["ACTIVE_AMT_ANNUITY_MAX"])))), ((data["CLOSED_MONTHS_BALANCE_MAX_MAX"])))))) * ((((data["NAME_INCOME_TYPE_Student"]) < (((((data["ACTIVE_AMT_ANNUITY_MAX"]) / 2.0)) / 2.0)))*1.)))) 
    v["i419"] = 0.089991*np.tanh(np.where(np.where(((data["ORGANIZATION_TYPE_Security"]) * (data["AMT_ANNUITY"]))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["OCCUPATION_TYPE_Security_staff"] )<0, ((data["ORGANIZATION_TYPE_Security"]) * (((data["AMT_ANNUITY"]) * (data["AMT_ANNUITY"])))), (-1.0*((data["ORGANIZATION_TYPE_Security"]))) )) 
    v["i420"] = 0.090001*np.tanh(np.where((((((data["AMT_INCOME_TOTAL"]) + (2.0))) > (1.570796))*1.)>0, (((3.141593) < (data["AMT_ANNUITY"]))*1.), ((((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) )) 
    v["i421"] = 0.099915*np.tanh((((((np.where(((data["ORGANIZATION_TYPE_XNA"]) / 2.0) < -99998, data["DAYS_EMPLOYED"], data["AMT_INCOME_TOTAL"] )) > ((((data["DAYS_EMPLOYED"]) > ((-1.0*((((((6.68204927444458008)) > (data["DAYS_EMPLOYED"]))*1.))))))*1.)))*1.)) * (data["DAYS_EMPLOYED"]))) 
    v["i422"] = 0.059958*np.tanh((-1.0*((((data["ORGANIZATION_TYPE_Transport__type_2"]) - ((((2.88201642036437988)) * (((((np.where(data["AMT_ANNUITY"]>0, data["ORGANIZATION_TYPE_Transport__type_2"], data["NEW_INC_PER_CHLD"] )) * (data["ORGANIZATION_TYPE_Transport__type_2"]))) * (data["AMT_ANNUITY"])))))))))) 
    v["i423"] = 0.083100*np.tanh((-1.0*((np.where(data["HOUSETYPE_MODE_block_of_flats"]>0, (((data["REG_REGION_NOT_LIVE_REGION"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["ORGANIZATION_TYPE_Telecom"]))/2.0)))/2.0), ((np.where(data["REG_REGION_NOT_LIVE_REGION"] < -99998, data["REG_REGION_NOT_LIVE_REGION"], data["NAME_TYPE_SUITE_Other_A"] )) - (data["ORGANIZATION_TYPE_Telecom"])) ))))) 
    v["i424"] = 0.038796*np.tanh(((((((0.85919755697250366)) < (np.where((((0.85919755697250366)) - (data["AMT_CREDIT"]))<0, np.where(data["AMT_CREDIT"] < -99998, 2.0, data["NEW_CREDIT_TO_ANNUITY_RATIO"] ), ((data["AMT_CREDIT"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])) )))*1.)) / 2.0)) 
    v["i425"] = 0.094983*np.tanh(np.minimum((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((data["FLAG_OWN_CAR"]) / 2.0), ((data["OCCUPATION_TYPE_Cooking_staff"]) * (data["AMT_INCOME_TOTAL"])) )))*1.))), ((((data["DAYS_BIRTH"]) * (((data["OCCUPATION_TYPE_Cooking_staff"]) / 2.0))))))) 
    v["i426"] = 0.064559*np.tanh(np.where(data["ORGANIZATION_TYPE_Industry__type_11"]<0, (((((1.0) / 2.0)) < (np.where(((data["OCCUPATION_TYPE_Cooking_staff"]) + (data["AMT_ANNUITY"]))<0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"])), data["ORGANIZATION_TYPE_Transport__type_2"] )))*1.), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) 
    v["i427"] = 0.099554*np.tanh(np.where(np.where(data["REG_CITY_NOT_WORK_CITY"]<0, (((data["ORGANIZATION_TYPE_Industry__type_11"]) < (data["AMT_CREDIT"]))*1.), data["AMT_CREDIT"] )<0, (((data["ORGANIZATION_TYPE_Industry__type_11"]) + (np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))/2.0), (((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.) )) 
    v["i428"] = 0.097400*np.tanh(((np.minimum(((np.maximum(((data["POS_SK_DPD_MAX"])), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["POS_SK_DPD_MAX"]))) - (data["POS_SK_DPD_MAX"]))))))), ((((((data["POS_SK_DPD_MEAN"]) - (data["POS_SK_DPD_MAX"]))) * 2.0))))) * 2.0)) 
    v["i429"] = 0.082300*np.tanh(((np.where(data["POS_SK_DPD_MAX"]<0, data["POS_SK_DPD_MAX"], ((3.0) * 2.0) )) * ((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)) > (np.where(data["AMT_ANNUITY"]>0, 0.318310, 1.0 )))*1.)) * 2.0)))) 
    v["i430"] = 0.087798*np.tanh((-1.0*((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, 1.570796, np.where((((((np.tanh(((2.21421408653259277)))) * 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)>0, (7.0), data["NAME_INCOME_TYPE_Student"] ) )) * ((10.0))))))) 
    v["i431"] = 0.062602*np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]>0, ((((13.09417152404785156)) + (((((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) - (data["NAME_INCOME_TYPE_State_servant"]))) * 2.0)) * 2.0)))/2.0), 1.570796 )))*1.)) * 2.0)) 
    v["i432"] = 0.099650*np.tanh((((((data["AMT_INCOME_TOTAL"]) < (np.minimum(((((((np.tanh((data["ORGANIZATION_TYPE_Telecom"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), (((((0.318310) < (data["AMT_INCOME_TOTAL"]))*1.))))))*1.)) * (data["OCCUPATION_TYPE_High_skill_tech_staff"]))) 
    v["i433"] = 0.099800*np.tanh((((((-1.0*(((13.76046848297119141))))) * ((((data["AMT_CREDIT"]) > (3.0))*1.)))) * (((data["LIVINGAPARTMENTS_MEDI"]) + ((((-1.0*((1.0)))) + (data["AMT_ANNUITY"]))))))) 
    v["i434"] = 0.096440*np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((3.141593) + (np.maximum(((((np.tanh((1.570796))) / 2.0))), (((((((6.62170648574829102)) / 2.0)) * (data["AMT_ANNUITY"])))))))/2.0)))*1.)) * 2.0)) * 2.0)) 
    v["i435"] = 0.069395*np.tanh(np.minimum(((((((((np.tanh((data["AMT_INCOME_TOTAL"]))) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) > ((((2.0) < (data["AMT_ANNUITY"]))*1.)))*1.))), ((((((((data["NAME_INCOME_TYPE_Student"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) < (data["AMT_ANNUITY"]))*1.))))) 
    v["i436"] = 0.096980*np.tanh((-1.0*((np.minimum(((np.maximum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((((data["NEW_CAR_TO_BIRTH_RATIO"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (((data["NEW_CAR_TO_BIRTH_RATIO"]) - (data["FLAG_OWN_CAR"]))))))))))) 
    v["i437"] = 0.097900*np.tanh((((((((data["NEW_CAR_TO_EMPLOY_RATIO"]) + (data["OWN_CAR_AGE"]))) > (np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]>0, (((0.318310) + (data["LIVE_CITY_NOT_WORK_CITY"]))/2.0), ((data["OWN_CAR_AGE"]) * 2.0) )))*1.)) * 2.0)) 
    v["i438"] = 0.065499*np.tanh(((((data["OWN_CAR_AGE"]) * ((((data["AMT_INCOME_TOTAL"]) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))))*1.)))) * ((((data["AMT_INCOME_TOTAL"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))) 
    v["i439"] = 0.087788*np.tanh(((((((((((-1.0) + (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (-2.0))/2.0)) + (-2.0))/2.0)))/2.0)) > (np.where(data["NEW_CAR_TO_BIRTH_RATIO"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["LIVE_CITY_NOT_WORK_CITY"] )))*1.)) * 2.0)) * 2.0)) 
    v["i440"] = 0.099998*np.tanh(np.where(((1.570796) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))<0, data["NEW_CAR_TO_BIRTH_RATIO"], np.minimum((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["FLAG_OWN_CAR"]))*1.))), ((((((data["NAME_INCOME_TYPE_Student"]) * ((11.31053447723388672)))) * (data["FLAG_OWN_CAR"]))))) )) 
    v["i441"] = 0.085490*np.tanh(((np.where(data["LIVINGAPARTMENTS_AVG"]<0, ((data["ORGANIZATION_TYPE_Government"]) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["ORGANIZATION_TYPE_Government"], data["NAME_INCOME_TYPE_Student"] ))), ((data["WEEKDAY_APPR_PROCESS_START_MONDAY"]) - (data["ORGANIZATION_TYPE_Government"])) )) * (((data["AMT_INCOME_TOTAL"]) * 2.0)))) 
    v["i442"] = 0.099900*np.tanh(((0.318310) * ((-1.0*((np.where(((-1.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))<0, data["NAME_INCOME_TYPE_Student"], np.where((((data["AMT_ANNUITY"]) + (data["LIVINGAPARTMENTS_MODE"]))/2.0) < -99998, 1.570796, data["WEEKDAY_APPR_PROCESS_START_MONDAY"] ) ))))))) 
    v["i443"] = 0.099970*np.tanh(np.where(data["NONLIVINGAPARTMENTS_MEDI"]>0, data["LIVE_CITY_NOT_WORK_CITY"], ((((((data["WALLSMATERIAL_MODE_Wooden"]) > (data["NONLIVINGAPARTMENTS_MEDI"]))*1.)) < (np.tanh(((((np.where(data["LIVINGAPARTMENTS_MEDI"]>0, data["LIVE_CITY_NOT_WORK_CITY"], data["NONLIVINGAPARTMENTS_MEDI"] )) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)))))*1.) )) 
    v["i444"] = 0.097640*np.tanh(np.where(data["NONLIVINGAPARTMENTS_AVG"] < -99998, 0.0, ((((data["AMT_ANNUITY"]) * (((((data["AMT_ANNUITY"]) * (data["NONLIVINGAPARTMENTS_AVG"]))) * (data["AMT_ANNUITY"]))))) * ((((data["AMT_ANNUITY"]) + (data["NONLIVINGAPARTMENTS_MEDI"]))/2.0))) )) 
    v["i445"] = 0.070959*np.tanh(((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < ((1.0)))*1.)) < (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["LIVE_CITY_NOT_WORK_CITY"]))*1.), ((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["LIVE_CITY_NOT_WORK_CITY"])))) )))*1.)) 
    v["i446"] = 0.078109*np.tanh(np.where(np.where(data["BURO_STATUS_C_MEAN_MEAN"]<0, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], ((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) * (data["BURO_STATUS_2_MEAN_MEAN"])) )>0, data["CLOSED_MONTHS_BALANCE_MIN_MIN"], np.tanh(((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))*1.))) )) 
    v["i447"] = 0.050001*np.tanh(np.minimum((((((data["BURO_STATUS_0_MEAN_MEAN"]) + ((-1.0*((data["CLOSED_MONTHS_BALANCE_MIN_MIN"])))))/2.0))), ((((((-1.0*((data["CLOSED_MONTHS_BALANCE_MIN_MIN"])))) < (((data["BURO_STATUS_0_MEAN_MEAN"]) + (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) * (data["BURO_STATUS_0_MEAN_MEAN"]))))))*1.))))) 
    v["i448"] = 0.099952*np.tanh(np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, ((data["BURO_STATUS_4_MEAN_MEAN"]) - (data["BURO_MONTHS_BALANCE_MAX_MAX"])), ((data["BURO_STATUS_0_MEAN_MEAN"]) * (np.where(data["BURO_MONTHS_BALANCE_MAX_MAX"]<0, ((data["BURO_STATUS_4_MEAN_MEAN"]) - (data["BURO_STATUS_0_MEAN_MEAN"])), data["BURO_STATUS_4_MEAN_MEAN"] ))) )) 
    v["i449"] = 0.075350*np.tanh((-1.0*(((((((((-1.0*((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))))) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))*1.)) < (np.tanh((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))))*1.))))) 
    v["i450"] = 0.098650*np.tanh(((np.where(data["AMT_ANNUITY"] < -99998, data["NAME_INCOME_TYPE_Student"], (((1.570796) < (((((((2.0) + (np.where(data["AMT_ANNUITY"]<0, data["CLOSED_MONTHS_BALANCE_MIN_MIN"], data["AMT_INCOME_TOTAL"] )))/2.0)) + (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))/2.0)))*1.) )) * 2.0)) 
    v["i451"] = 0.066996*np.tanh(((((((np.where(np.where(data["BURO_STATUS_2_MEAN_MEAN"]<0, (-1.0*((data["BURO_STATUS_2_MEAN_MEAN"]))), ((-2.0) + (data["BURO_STATUS_2_MEAN_MEAN"])) )<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0*((data["NAME_INCOME_TYPE_Student"]))) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i452"] = 0.094900*np.tanh(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) * ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.maximum((((((((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((2.93883275985717773)))*1.))), ((data["AMT_ANNUITY"])))))) > ((2.93883275985717773)))*1.)))) 
    v["i453"] = 0.095479*np.tanh(((((((((((((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) * (data["BURO_STATUS_C_MEAN_MEAN"]))) * 2.0)) + (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))/2.0)) < (((data["BURO_STATUS_4_MEAN_MEAN"]) - (3.141593))))*1.)) * 2.0)) * 2.0)) 
    v["i454"] = 0.030401*np.tanh((-1.0*(((((np.where(data["BURO_STATUS_2_MEAN_MEAN"]>0, ((-2.0) / 2.0), np.where((((1.570796) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))*1.)>0, ((-2.0) / 2.0), (1.74125826358795166) ) )) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))*1.))))) 
    v["i455"] = 0.032510*np.tanh((-1.0*(((((((((1.87832641601562500)) < (data["BURO_STATUS_2_MEAN_MEAN"]))*1.)) < (np.where(data["BURO_STATUS_2_MEAN_MEAN"]>0, data["BURO_MONTHS_BALANCE_MAX_MAX"], (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > ((((2.0) + (1.570796))/2.0)))*1.) )))*1.))))) 
    v["i456"] = 0.096860*np.tanh(((data["AMT_ANNUITY"]) * ((((((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (1.570796))) < (data["BURO_MONTHS_BALANCE_MAX_MAX"]))*1.)) + (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, 2.0, (((1.570796) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.) )))))) 
    v["i457"] = 0.099244*np.tanh((-1.0*(((((np.where((((data["BURO_STATUS_0_MEAN_MEAN"]) + (np.tanh((data["BURO_STATUS_2_MEAN_MEAN"]))))/2.0)>0, np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]>0, data["BURO_STATUS_0_MEAN_MEAN"], data["BURO_STATUS_2_MEAN_MEAN"] ), 3.0 )) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.))))) 
    v["i458"] = 0.096120*np.tanh((((((((((0.636620) > (((3.141593) - (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))))*1.)) * 2.0)) - (np.tanh(((((((2.0) - (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))) < (0.636620))*1.)))))) * 2.0)) 
    v["i459"] = 0.095590*np.tanh(np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_Telecom"])), (((((((data["AMT_ANNUITY"]) > (3.141593))*1.)) * (data["AMT_ANNUITY"]))))))), ((np.where(data["AMT_ANNUITY"]>0, data["ORGANIZATION_TYPE_Transport__type_1"], data["ORGANIZATION_TYPE_Telecom"] ))))) 
    v["i460"] = 0.099833*np.tanh(np.minimum((((((((((((data["NAME_INCOME_TYPE_Student"]) + ((-1.0*((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))/2.0)) * (data["ORGANIZATION_TYPE_Construction"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["ORGANIZATION_TYPE_Transport__type_1"])))), ((data["ORGANIZATION_TYPE_Transport__type_1"])))) 
    v["i461"] = 0.044999*np.tanh(((((((((((np.minimum((((-1.0*((data["OCCUPATION_TYPE_Secretaries"]))))), ((((data["NAME_TYPE_SUITE_Other_B"]) * ((-1.0*((data["NAME_TYPE_SUITE_Other_B"]))))))))) * (data["AMT_ANNUITY"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i462"] = 0.036997*np.tanh(np.where(data["NAME_EDUCATION_TYPE_Lower_secondary"]>0, data["OCCUPATION_TYPE_Cooking_staff"], (((-1.0*(((((data["OCCUPATION_TYPE_Cooking_staff"]) < (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["OCCUPATION_TYPE_Cooking_staff"], np.minimum(((data["ORGANIZATION_TYPE_Industry__type_12"])), ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0)))) )))*1.))))) * 2.0) )) 
    v["i463"] = 0.097979*np.tanh((-1.0*((np.where(np.where(data["AMT_ANNUITY"]<0, (((data["OCCUPATION_TYPE_Cooking_staff"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.), data["OCCUPATION_TYPE_Cooking_staff"] )>0, np.where(data["AMT_INCOME_TOTAL"]>0, data["OCCUPATION_TYPE_Cooking_staff"], (-1.0*((data["NAME_INCOME_TYPE_Maternity_leave"]))) ), data["ORGANIZATION_TYPE_Industry__type_12"] ))))) 
    v["i464"] = 0.097701*np.tanh(((data["WALLSMATERIAL_MODE_Wooden"]) * (((np.where(data["WALLSMATERIAL_MODE_Wooden"]>0, np.where(data["AMT_ANNUITY"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_ANNUITY"] ), data["WALLSMATERIAL_MODE_Mixed"] )) * ((((data["AMT_ANNUITY"]) + ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))/2.0)))))) 
    v["i465"] = 0.087593*np.tanh(((data["FONDKAPREMONT_MODE_not_specified"]) * (((data["FONDKAPREMONT_MODE_not_specified"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.where(np.where(data["AMT_INCOME_TOTAL"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NEW_ANNUITY_TO_INCOME_RATIO"] )>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["AMT_INCOME_TOTAL"] )))))))) 
    v["i466"] = 0.019960*np.tanh(np.minimum((((((data["WALLSMATERIAL_MODE_Wooden"]) > (((((data["AMT_ANNUITY"]) / 2.0)) / 2.0)))*1.))), ((np.where(data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]<0, ((((-1.0*((data["OCCUPATION_TYPE_Drivers"])))) < (data["OCCUPATION_TYPE_Drivers"]))*1.), (-1.0*((data["OCCUPATION_TYPE_Drivers"]))) ))))) 
    v["i467"] = 0.046600*np.tanh((((((((3.141593) + ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["AMT_ANNUITY"]) + (data["OCCUPATION_TYPE_Drivers"]))))*1.)))) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["AMT_ANNUITY"]))))/2.0)))*1.)) * 2.0)) 
    v["i468"] = 0.067994*np.tanh((-1.0*((np.where(data["WALLSMATERIAL_MODE_Wooden"]<0, (((((0.318310) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0)))/2.0)) * (data["ORGANIZATION_TYPE_Culture"])), (((data["NAME_INCOME_TYPE_Student"]) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (0.318310))/2.0)))*1.) ))))) 
    v["i469"] = 0.077995*np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["AMT_INCOME_TOTAL"]))*1.)) * (((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["AMT_INCOME_TOTAL"]))*1.)) < (data["AMT_INCOME_TOTAL"]))*1.)) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) 
    v["i470"] = 0.099770*np.tanh(((((data["ORGANIZATION_TYPE_Culture"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.minimum(((np.where(((data["ORGANIZATION_TYPE_Culture"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["OCCUPATION_TYPE_Drivers"] ))), ((data["NAME_EDUCATION_TYPE_Lower_secondary"])))))))) * (data["NAME_EDUCATION_TYPE_Lower_secondary"]))) 
    v["i471"] = 0.091999*np.tanh((-1.0*((((((np.where(((data["AMT_INCOME_TOTAL"]) + (((0.318310) + (data["NAME_INCOME_TYPE_Student"]))))<0, data["NAME_EDUCATION_TYPE_Lower_secondary"], np.where(data["AMT_INCOME_TOTAL"]<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NAME_EDUCATION_TYPE_Lower_secondary"] ) )) / 2.0)) / 2.0))))) 
    v["i472"] = 0.083551*np.tanh(((data["OCCUPATION_TYPE_Drivers"]) * (np.where(((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) + (np.minimum(((data["AMT_ANNUITY"])), (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))*1.))))))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((-1.0) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))*1.) )))) 
    v["i473"] = 0.087699*np.tanh((((6.01758050918579102)) * ((((6.01758050918579102)) * ((((6.01758050918579102)) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * ((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (((((6.01758050918579102)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))*1.)))))))))) 
    v["i474"] = 0.099500*np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (2.0))) / 2.0), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (2.0)) )))*1.)) * (data["CODE_GENDER"]))) 
    v["i475"] = 0.097000*np.tanh((((((np.where(0.636620>0, (((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) * (data["AMT_INCOME_TOTAL"]))) + (((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) * (((data["AMT_INCOME_TOTAL"]) * 2.0)))))/2.0), 0.636620 )) > (1.0))*1.)) * 2.0)) 
    v["i476"] = 0.019800*np.tanh(np.maximum(((np.where(data["POS_NAME_CONTRACT_STATUS_Active_MEAN"]>0, (((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) + (data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"]))/2.0), (((-1.0*((0.636620)))) / 2.0) ))), ((np.where(data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]<0, data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"], data["POS_NAME_CONTRACT_STATUS_Active_MEAN"] ))))) 
    v["i477"] = 0.029940*np.tanh(np.where(data["POS_SK_DPD_MEAN"]<0, np.maximum((((((data["POS_SK_DPD_MEAN"]) > (((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) / 2.0)))*1.))), (((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) / 2.0)) > (3.141593))*1.)))), (((3.0) < (data["POS_SK_DPD_MAX"]))*1.) )) 
    v["i478"] = 0.099999*np.tanh((((((((data["POS_SK_DPD_DEF_MEAN"]) > (np.maximum(((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"])), (((-1.0*((data["POS_SK_DPD_MAX"]))))))))*1.)) * 2.0)) * (np.where(data["POS_SK_DPD_DEF_MEAN"]>0, (4.33465480804443359), data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"] )))) 
    v["i479"] = 0.042010*np.tanh((((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) > (np.where(data["POS_NAME_CONTRACT_STATUS_Active_MEAN"]>0, (((np.minimum(((data["POS_SK_DPD_MEAN"])), ((data["POS_NAME_CONTRACT_STATUS_Active_MEAN"])))) > (data["POS_SK_DPD_MAX"]))*1.), (10.13535594940185547) )))*1.)) + (((np.tanh((data["POS_SK_DPD_MEAN"]))) / 2.0)))) 
    v["i480"] = 0.098000*np.tanh(np.where(data["INSTAL_PAYMENT_PERC_MEAN"]>0, np.where((-1.0*((data["INSTAL_PAYMENT_PERC_MAX"])))<0, data["INSTAL_DBD_MAX"], -2.0 ), np.where(data["INSTAL_PAYMENT_PERC_MAX"]<0, (((data["INSTAL_DBD_MAX"]) > ((1.69303214550018311)))*1.), 2.0 ) )) 
    v["i481"] = 0.099996*np.tanh((-1.0*((np.where((((data["REFUSED_CNT_PAYMENT_MEAN"]) < (((data["AMT_ANNUITY"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))*1.)>0, 0.0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((((data["REFUSED_CNT_PAYMENT_MEAN"]) * 2.0)))))) ))))) 
    v["i482"] = 0.099910*np.tanh((((((((3.141593) < (data["AMT_CREDIT"]))*1.)) * (np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] < -99998, ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (3.141593)), (4.74292945861816406) )))) * ((4.74292945861816406)))) 
    v["i483"] = 0.056399*np.tanh((-1.0*(((((-1.0) > (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, ((((data["AMT_ANNUITY"]) * (np.maximum(((-1.0)), ((data["AMT_ANNUITY"])))))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), data["NEW_CREDIT_TO_INCOME_RATIO"] )))*1.))))) 
    v["i484"] = 0.097201*np.tanh((((((((np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]>0, 0.0, 2.0 )) < (((np.where((-1.0*((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_INCOME_TOTAL"] )) * (data["CODE_GENDER"]))))*1.)) * 2.0)) * 2.0)) 
    v["i485"] = 0.013002*np.tanh(np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, 0.0, np.minimum((((((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) < (((data["BURO_MONTHS_BALANCE_MAX_MAX"]) * 2.0)))*1.)) - ((-1.0*((data["BURO_STATUS_2_MEAN_MEAN"]))))))), (((((data["BURO_STATUS_4_MEAN_MEAN"]) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))*1.)))) )) 
    v["i486"] = 0.099849*np.tanh(((((((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) < (((((((-1.0) + (data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))/2.0)) > (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))*1.)))*1.)) < (data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))*1.)) * ((-1.0*(((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) < (data["BURO_MONTHS_BALANCE_SIZE_MEAN"]))*1.))))))) 
    v["i487"] = 0.099602*np.tanh(((np.minimum((((((((3.0) / 2.0)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.))), ((np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (0.318310))*1.))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))))))) / 2.0)) 
    v["i488"] = 0.098904*np.tanh((((((((data["INSTAL_PAYMENT_PERC_MAX"]) * 2.0)) > (((data["INSTAL_PAYMENT_PERC_MEAN"]) + (data["NAME_INCOME_TYPE_Student"]))))*1.)) - ((((data["INSTAL_PAYMENT_PERC_MAX"]) > (((((data["NAME_INCOME_TYPE_Student"]) * (0.636620))) + (data["NAME_INCOME_TYPE_Student"]))))*1.)))) 
    v["i489"] = 0.100000*np.tanh((((((-1.0) > (((np.maximum(((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) * ((5.0))))), ((np.maximum(((0.636620)), ((((data["NAME_INCOME_TYPE_Student"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))) - (data["INSTAL_AMT_INSTALMENT_MAX"]))))*1.)) * 2.0)) 
    v["i490"] = 0.081300*np.tanh(((2.0) * ((-1.0*(((((np.where(data["AMT_ANNUITY"]<0, data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"], data["CC_AMT_DRAWINGS_CURRENT_VAR"] )) > (np.where(data["AMT_ANNUITY"]<0, (((data["AMT_ANNUITY"]) < (data["CC_AMT_DRAWINGS_CURRENT_VAR"]))*1.), data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"] )))*1.))))))) 
    v["i491"] = 0.084964*np.tanh((((3.141593) < (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]>0, np.where(((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]) + (((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) * 2.0)))>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], (14.65441703796386719) ), ((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]) + (1.0)) )))*1.)) 
    v["i492"] = 0.084899*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]<0, ((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((((np.where(data["NAME_HOUSING_TYPE_Rented_apartment"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] )) * (data["REG_CITY_NOT_LIVE_CITY"])))))) * (np.tanh((data["NAME_HOUSING_TYPE_Rented_apartment"])))), data["NAME_HOUSING_TYPE_Rented_apartment"] )) 
    v["i493"] = 0.099903*np.tanh(((((((((-1.0*(((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (data["NAME_HOUSING_TYPE_Rented_apartment"]))*1.))))) < (data["LIVINGAREA_MODE"]))*1.)) * (np.where((((data["LIVINGAREA_MEDI"]) + (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))/2.0)<0, data["LIVINGAREA_MODE"], data["NAME_HOUSING_TYPE_Rented_apartment"] )))) * 2.0)) 
    v["i494"] = 0.079029*np.tanh((((((0.318310) < (np.where(data["LIVINGAREA_MEDI"] < -99998, data["FLOORSMAX_MODE"], (((((data["LIVINGAREA_MEDI"]) < (np.where(data["LIVINGAREA_MEDI"]>0, data["FLOORSMAX_MODE"], data["LIVINGAREA_AVG"] )))*1.)) - (data["LIVINGAREA_AVG"])) )))*1.)) / 2.0)) 
    v["i495"] = 0.093600*np.tanh(np.where(data["NEW_INC_BY_ORG"]<0, ((np.where(data["AMT_ANNUITY"]>0, data["REG_CITY_NOT_LIVE_CITY"], np.where(data["OCCUPATION_TYPE_Laborers"]>0, data["REG_CITY_NOT_LIVE_CITY"], data["NAME_INCOME_TYPE_Student"] ) )) * 2.0), (-1.0*(((((data["REG_CITY_NOT_LIVE_CITY"]) > (data["NEW_INC_BY_ORG"]))*1.)))) )) 
    v["i496"] = 0.097989*np.tanh(np.where((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) > (((0.318310) * 2.0)))*1.)>0, (((((data["INSTAL_PAYMENT_PERC_MAX"]) > (data["INSTAL_PAYMENT_PERC_MEAN"]))*1.)) * 2.0), np.maximum((((((data["INSTAL_PAYMENT_PERC_MEAN"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]))/2.0))), ((data["NAME_HOUSING_TYPE_Rented_apartment"]))) )) 
    v["i497"] = 0.094620*np.tanh((-1.0*(((((((((2.0) < (np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]<0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]<0, np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])), ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))), data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"] ) )))*1.)) * 2.0)) * 2.0))))) 
    v["i498"] = 0.099922*np.tanh((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) > (((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) + (np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]<0, (((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]) < (data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"]))*1.), (((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) > (data["CC_AMT_DRAWINGS_OTHER_CURRENT_MAX"]))*1.) )))))*1.)) 
    v["i499"] = 0.090654*np.tanh(((((np.minimum(((np.where(data["AMT_ANNUITY"]<0, data["NAME_EDUCATION_TYPE_Higher_education"], np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["OCCUPATION_TYPE_Laborers"], data["AMT_ANNUITY"] ) ))), ((np.where(data["AMT_CREDIT"]<0, data["OCCUPATION_TYPE_Laborers"], data["NAME_EDUCATION_TYPE_Higher_education"] ))))) / 2.0)) / 2.0)) 
    v["i500"] = 0.099320*np.tanh(((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0)) * 2.0)) * 2.0)) * ((-1.0*(((((np.where(np.minimum(((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN"])), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])))>0, 3.141593, data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] )) > (2.0))*1.))))))) 
    v["i501"] = 0.100000*np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.where(data["AMT_ANNUITY"]>0, ((0.636620) + (data["AMT_ANNUITY"])), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )))) > (3.0))*1.)) - (((data["ORGANIZATION_TYPE_Industry__type_2"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) 
    v["i502"] = 0.098000*np.tanh((-1.0*(((((((((data["ORGANIZATION_TYPE_Industry__type_2"]) + (data["ORGANIZATION_TYPE_Industry__type_2"]))) + ((((data["NAME_INCOME_TYPE_Student"]) + (data["ORGANIZATION_TYPE_Industry__type_2"]))/2.0)))) > ((-1.0*((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (2.0)))))))*1.))))) 
    v["i503"] = 0.096000*np.tanh((((((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * (np.where(((((2.13039326667785645)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)>0, (((7.0)) * (np.minimum(((data["NONLIVINGAREA_MODE"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))), (6.0) )))) 
    v["i504"] = 0.098781*np.tanh(((((((((((4.0)) + ((((data["AMT_ANNUITY"]) > (((((-1.0*((data["NONLIVINGAREA_MEDI"])))) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)))*1.)))/2.0)) + (1.570796))/2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) 
    v["i505"] = 0.010101*np.tanh((((((np.where(data["NONLIVINGAREA_AVG"]>0, ((np.minimum(((((0.636620) + (0.318310)))), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) - (data["NONLIVINGAREA_MODE"])), data["NONLIVINGAREA_AVG"] )) - (data["NONLIVINGAREA_MODE"]))) > (0.318310))*1.)) 
    v["i506"] = 0.099900*np.tanh(((3.141593) * (((3.141593) * (((((((2.79250931739807129)) < (data["LIVINGAPARTMENTS_MEDI"]))*1.)) * (((data["AMT_ANNUITY"]) * (((data["NONLIVINGAREA_MODE"]) - ((10.83971786499023438)))))))))))) 
    v["i507"] = 0.097990*np.tanh((((((((data["LIVINGAPARTMENTS_MEDI"]) > (((3.0) + (np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NONLIVINGAREA_AVG"])))))/2.0)<0, data["NONLIVINGAREA_MODE"], data["LIVINGAPARTMENTS_MEDI"] )))))*1.)) * 2.0)) * 2.0)) 
    v["i508"] = 0.098996*np.tanh((-1.0*(((((0.636620) < (np.where(data["LIVINGAPARTMENTS_MEDI"]>0, data["LIVINGAPARTMENTS_MODE"], (((((np.maximum(((data["LIVINGAPARTMENTS_MEDI"])), ((data["NONLIVINGAPARTMENTS_MODE"])))) + ((((data["LIVINGAPARTMENTS_MEDI"]) > (data["NONLIVINGAPARTMENTS_MODE"]))*1.)))/2.0)) * 2.0) )))*1.))))) 
    v["i509"] = 0.056190*np.tanh((((((((data["HOUSETYPE_MODE_block_of_flats"]) > (np.where(np.where(data["LIVINGAREA_AVG"] < -99998, 3.0, data["HOUSETYPE_MODE_block_of_flats"] )<0, ((data["LIVINGAREA_MODE"]) * 2.0), 3.0 )))*1.)) - (data["NAME_INCOME_TYPE_Student"]))) * 2.0)) 
    v["i510"] = 0.095000*np.tanh((((((((data["FLOORSMIN_AVG"]) > (((data["HOUSETYPE_MODE_block_of_flats"]) + ((((data["FLOORSMIN_MEDI"]) + (np.where(data["LIVINGAREA_MODE"] < -99998, data["HOUSETYPE_MODE_block_of_flats"], np.tanh((np.tanh((-1.0)))) )))/2.0)))))*1.)) * 2.0)) * 2.0)) 
    v["i511"] = 0.056400*np.tanh(((data["HOUSETYPE_MODE_specific_housing"]) * (((((np.where(data["WALLSMATERIAL_MODE_Panel"]<0, (-1.0*((np.where(data["HOUSETYPE_MODE_specific_housing"]<0, data["FONDKAPREMONT_MODE_reg_oper_account"], data["HOUSETYPE_MODE_specific_housing"] )))), data["FONDKAPREMONT_MODE_reg_oper_account"] )) + (data["ORGANIZATION_TYPE_Trade__type_4"]))) + (data["NAME_INCOME_TYPE_Student"])))))
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


def prepare_gp1_features():
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
    train = GP1(train)
    test = GP1(test)
    train['TARGET'] = traintargets
    
    del train["TARGET"]
    train.to_csv("processed/train_gp1_features.csv", index = False)
    test.to_csv("processed/test_gp1_features.csv", index = False)