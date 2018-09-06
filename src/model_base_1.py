# This script was redacted by Ramzi Bouyekhf (propower)

import pandas as pd 
import numpy as np
import datetime
import gc


def prepare_train_test():
    #load dataframes
    application_train = pd.read_csv("input/application_train.csv").set_index("SK_ID_CURR")
    application_test = pd.read_csv("input/application_test.csv").set_index("SK_ID_CURR")
    previous_application = pd.read_csv("input/previous_application.csv")
    bureau = pd.read_csv("input/bureau.csv")
    bureau_balance = pd.read_csv("input/bureau_balance.csv") 
    POS_CASH_balance = pd.read_csv("input/POS_CASH_balance.csv")
    credit_card_balance = pd.read_csv("input/credit_card_balance.csv")
    installments_payments = pd.read_csv("input/installments_payments.csv")
    
    #prepare bureau balance
    bureau_balance["SK_ID_CURR"] = bureau_balance["SK_ID_BUREAU"].map(bureau.set_index("SK_ID_BUREAU")["SK_ID_CURR"])
    bureau_balance = bureau_balance.dropna(subset = ["SK_ID_CURR"])
    bureau_balance["SK_ID_CURR"] = bureau_balance["SK_ID_CURR"].astype(int)
    bureau_balance.head()
    
    #prepapre application
    application = application_train.append(application_test)
    
    #add number of problems for each active credit detected in Credit Bureau reports. Credit is considered active when CREDIT_ACTIVE == Active. Problem is detected when  AMT_CREDIT_MAX_OVERDUE > 0 (the max overdue is the amount of money the borrower didn't pay in time) an or DAYS_CREDIT_ENDDATE < 0 (which means the credit should be closed but it's not).
    bureau_credits = bureau.copy()
    bureau_credits["curr_prb_bur_active"] = 0
    bureau_credits["curr_prb_bur_active"] = (((bureau_credits["DAYS_CREDIT_ENDDATE"] < 0) & (bureau_credits["AMT_CREDIT_SUM"] > 0)) 
                                      | (bureau["AMT_CREDIT_MAX_OVERDUE"] > 0)) & (bureau_credits["CREDIT_ACTIVE"] == "Active")
    bureau_credits = bureau_credits.groupby("SK_ID_CURR").agg({"curr_prb_bur_active" : "sum"})
    application = application.join(bureau_credits)
    del bureau_credits
    gc.collect()
    
    #add feature that describes the number of months the client was in DPD (Days Past Due) for the last 6 months using Credit Bureau balance table
    bureau_balance_current = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] >= -6].copy()
    bureau_balance_current["prbs_six_months_bur"] = 0
    bureau_balance_current.loc[(bureau_balance_current["STATUS"].isin(['1', '2', '3', '4', '5'])), "prbs_six_months_bur"] = 1
    bureau_balance_current = bureau_balance_current.groupby("SK_ID_CURR").agg({"prbs_six_months_bur" : "sum"})
    application = application.join(bureau_balance_current)
    del bureau_balance_current
    gc.collect()
    
    #add feature that describes the number of months the client was in DPD (Days Past Due) for the last 12 months using Credit Bureau balance table
    bureau_balance_current = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] >= -12].copy()
    bureau_balance_current["prbs_twelve_months_bur"] = 0
    bureau_balance_current.loc[(bureau_balance_current["STATUS"].isin(['1', '2', '3', '4', '5'])), "prbs_twelve_months_bur"] = 1
    bureau_balance_current = bureau_balance_current.groupby("SK_ID_CURR").agg({"prbs_twelve_months_bur" : "sum"})
    application = application.join(bureau_balance_current)
    del bureau_balance_current
    gc.collect()

    #add feature that describes the number of months the client was in DPD (Days Past Due) for the last month using Credit Bureau balance table
    bureau_balance_current = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] >= -1].copy()
    bureau_balance_current["prbs_one_months_bur"] = 0
    bureau_balance_current.loc[(bureau_balance_current["STATUS"].isin(['1', '2', '3', '4', '5'])), "prbs_one_months_bur"] = 1
    bureau_balance_current = bureau_balance_current.groupby("SK_ID_CURR").agg({"prbs_one_months_bur" : "sum"})
    application = application.join(bureau_balance_current)
    del bureau_balance_current
    gc.collect()
    
    #did clients have previous credits or not ?
    ids_history = np.unique(np.append(bureau["SK_ID_CURR"].unique(), previous_application["SK_ID_CURR"].unique()))
    ids_history = np.unique(np.append(ids_history, credit_card_balance["SK_ID_CURR"].unique()))
    ids_history = np.unique(np.append(ids_history, installments_payments["SK_ID_CURR"].unique()))
    ids_history = np.unique(np.append(ids_history, POS_CASH_balance["SK_ID_CURR"].unique()))
    application["has_prev_credit"] = np.where(application.reset_index()["SK_ID_CURR"].isin(ids_history), 1, 0)
    del ids_history
    gc.collect()
    
    #bureau not active credits. Did client have problems paying past bureau credits ? if yes, what's the ratio between the minimal amount credit that encountered problems compared to actual amount demanded in application ?
    bureau_past = bureau.loc[bureau["CREDIT_ACTIVE"] != "Active"]
    bureau_past["difficulties_past"] = ((bureau_past["DAYS_ENDDATE_FACT"] - bureau_past["DAYS_CREDIT_ENDDATE"] > 30) & bureau_past["AMT_CREDIT_SUM"] > 0)| (bureau_past["AMT_CREDIT_MAX_OVERDUE"] > 0)
    bureau_past["amt_bur_credit_diff"] = np.where(bureau_past["difficulties_past"] > 0, bureau_past["AMT_CREDIT_SUM"], 0)
    bureau_past = bureau_past.groupby("SK_ID_CURR").agg({"difficulties_past" : "sum", "amt_bur_credit_diff" : "min"})
    application = application.join(bureau_past)
    application["perc_min_bur_diff_amt_crd"] = application["amt_bur_credit_diff"]/application["AMT_CREDIT"]
    del bureau_past
    gc.collect()
    
    #How much problems did the clients have in paying debts that are greater than actual amount demanded in application ?
    bureau_balance_overall = bureau_balance.copy()
    bureau_balance_overall["amt_curr_credit"] = bureau_balance_overall["SK_ID_CURR"].map(application["AMT_CREDIT"])
    bureau_balance_overall["amt_bur_credit"] = bureau_balance_overall["SK_ID_BUREAU"].map(bureau.set_index("SK_ID_BUREAU")["AMT_CREDIT_SUM"])
    
    bureau_balance_overall["perc_curr_bur_credit"] = bureau_balance_overall["amt_curr_credit"]/bureau_balance_overall["amt_bur_credit"]
    bureau_balance_overall["curr_lt_bur"] = (bureau_balance_overall["amt_curr_credit"] < bureau_balance_overall["amt_bur_credit"])
    bureau_balance_overall["prb_bur"] = False
    bureau_balance_overall.loc[(bureau_balance_overall["STATUS"].isin(['1', '2', '3', '4', '5'])), "prb_bur"] = True
    bureau_balance_overall["prb_and_lt_curr"] = (bureau_balance_overall["curr_lt_bur"] & bureau_balance_overall["prb_bur"])
    bureau_balance_overall = bureau_balance_overall.groupby("SK_ID_CURR").agg({"perc_curr_bur_credit" : "max", "prb_and_lt_curr" : "sum",
                                                                              "prb_bur" : "sum"})
    application = application.join(bureau_balance_overall)
    del bureau_balance_overall
    gc.collect()
    
    #add number of problems in paying credit card credit for last 6 months
    prbs_credit_card_6 = credit_card_balance.loc[(credit_card_balance["MONTHS_BALANCE"] >= -6)].copy()
    prbs_credit_card_6["prbs_six_months"] = 0
    prbs_credit_card_6.loc[(prbs_credit_card_6["SK_DPD"] > 0) 
                             | (prbs_credit_card_6["AMT_PAYMENT_TOTAL_CURRENT"] > prbs_credit_card_6["AMT_INST_MIN_REGULARITY"])
                             | (prbs_credit_card_6["AMT_BALANCE"] < prbs_credit_card_6["AMT_INST_MIN_REGULARITY"])
                             | (prbs_credit_card_6["AMT_TOTAL_RECEIVABLE"] < prbs_credit_card_6["AMT_BALANCE"]), "prbs_six_months"] = 1
    prbs_credit_card_6["perc_paym_inst_min"] = prbs_credit_card_6["AMT_PAYMENT_TOTAL_CURRENT"]/prbs_credit_card_6["AMT_INST_MIN_REGULARITY"]
    prbs_credit_card_6["perc_inst_min_bal"] = prbs_credit_card_6["AMT_INST_MIN_REGULARITY"]/prbs_credit_card_6["AMT_BALANCE"]
    prbs_credit_card_6["perc_bal_tot_rec"] = prbs_credit_card_6["AMT_BALANCE"]/prbs_credit_card_6["AMT_TOTAL_RECEIVABLE"]
    prbs_credit_card_6 = prbs_credit_card_6.groupby("SK_ID_CURR").agg({"prbs_six_months" : "sum", "perc_paym_inst_min" :"min", 
                                                                       "perc_inst_min_bal" :"min", "perc_bal_tot_rec": "min"})
    application = application.join(prbs_credit_card_6)
    del prbs_credit_card_6
    gc.collect()
    
    #add number of problems in paying credit card credit for last 12 months
    prbs_credit_card_12 = credit_card_balance.loc[(credit_card_balance["MONTHS_BALANCE"] >= -12)].copy()
    prbs_credit_card_12["prbs_twelve_months"] = 0
    prbs_credit_card_12.loc[(prbs_credit_card_12["SK_DPD"] > 0) 
                             | (prbs_credit_card_12["AMT_PAYMENT_TOTAL_CURRENT"] > prbs_credit_card_12["AMT_INST_MIN_REGULARITY"])
                             | (prbs_credit_card_12["AMT_BALANCE"] < prbs_credit_card_12["AMT_INST_MIN_REGULARITY"])
                             | (prbs_credit_card_12["AMT_TOTAL_RECEIVABLE"] < prbs_credit_card_12["AMT_BALANCE"]), "prbs_twelve_months"] = 1
    prbs_credit_card_12["perc_paym_inst_min_12"] = prbs_credit_card_12["AMT_PAYMENT_TOTAL_CURRENT"]/prbs_credit_card_12["AMT_INST_MIN_REGULARITY"]
    prbs_credit_card_12["perc_inst_min_bal_12"] = prbs_credit_card_12["AMT_INST_MIN_REGULARITY"]/prbs_credit_card_12["AMT_BALANCE"]
    prbs_credit_card_12["perc_bal_tot_rec_12"] = prbs_credit_card_12["AMT_BALANCE"]/prbs_credit_card_12["AMT_TOTAL_RECEIVABLE"]
    prbs_credit_card_12 = prbs_credit_card_12.groupby("SK_ID_CURR").agg({"prbs_twelve_months" : "sum", "perc_paym_inst_min_12" :"min", 
                                                                       "perc_inst_min_bal_12" :"min", "perc_bal_tot_rec_12": "min"})
    application = application.join(prbs_credit_card_12)
    del prbs_credit_card_12
    gc.collect()
    
    #add number of problems in paying credit card credit for last 3 months
    prbs_credit_card_3 = credit_card_balance.loc[(credit_card_balance["MONTHS_BALANCE"] >= -3)].copy()
    prbs_credit_card_3["prbs_three_months"] = 0
    prbs_credit_card_3.loc[(prbs_credit_card_3["SK_DPD"] > 0) 
                             | (prbs_credit_card_3["AMT_PAYMENT_TOTAL_CURRENT"] > prbs_credit_card_3["AMT_INST_MIN_REGULARITY"])
                             | (prbs_credit_card_3["AMT_BALANCE"] < prbs_credit_card_3["AMT_INST_MIN_REGULARITY"])
                             | (prbs_credit_card_3["AMT_TOTAL_RECEIVABLE"] < prbs_credit_card_3["AMT_BALANCE"]), "prbs_three_months"] = 1
    prbs_credit_card_3["perc_paym_inst_min_3"] = prbs_credit_card_3["AMT_PAYMENT_TOTAL_CURRENT"]/prbs_credit_card_3["AMT_INST_MIN_REGULARITY"]
    prbs_credit_card_3["perc_inst_min_bal_3"] = prbs_credit_card_3["AMT_INST_MIN_REGULARITY"]/prbs_credit_card_3["AMT_BALANCE"]
    prbs_credit_card_3["perc_bal_tot_rec_3"] = prbs_credit_card_3["AMT_BALANCE"]/prbs_credit_card_3["AMT_TOTAL_RECEIVABLE"]
    prbs_credit_card_3 = prbs_credit_card_3.groupby("SK_ID_CURR").agg({"prbs_three_months" : "sum", "perc_paym_inst_min_3" :"min", 
                                                                       "perc_inst_min_bal_3" :"min", "perc_bal_tot_rec_3": "min"})
    application = application.join(prbs_credit_card_3)
    del prbs_credit_card_3
    gc.collect()
    
    # How many months client got DPD > 0 or DPD_DEF > 0 (DPD with tolerance) ?
    POS_CASH_balance = POS_CASH_balance.set_index("SK_ID_PREV").join(previous_application[["AMT_ANNUITY", "AMT_CREDIT", "SK_ID_PREV"]].set_index("SK_ID_PREV")).reset_index()
    pos_difficulties = POS_CASH_balance.copy()
    pos_difficulties["prbs_pos_dpd_def"] = 0
    pos_difficulties.loc[pos_difficulties["SK_DPD_DEF"] > 0, "prbs_pos_dpd_def"] = 1
    pos_difficulties["prbs_pos_dpd"] = 0
    pos_difficulties.loc[pos_difficulties["SK_DPD"] > 0, "prbs_pos_dpd"] = 1
    pos_difficulties = pos_difficulties.groupby("SK_ID_CURR").agg({"AMT_CREDIT" : "min", "prbs_pos_dpd" : "sum", "prbs_pos_dpd_def" : "sum"})
    application = application.join(pos_difficulties, rsuffix = "_POS")
    del pos_difficulties
    gc.collect()
    
    # How many times client paid less than 90% the minimal amount of the instalment in the last year ?
    inst_pay = installments_payments.copy()
    inst_pay["prb_insts_12"] = 0
    inst_pay["perc_paym_inst"] = inst_pay["AMT_PAYMENT"]/inst_pay["AMT_INSTALMENT"]
    inst_pay.loc[((inst_pay["AMT_PAYMENT"]/inst_pay["AMT_INSTALMENT"]) < 0.9) & (inst_pay["DAYS_INSTALMENT"] > -365), "prb_insts_12"] = 1
    inst_pay = inst_pay.groupby("SK_ID_CURR").agg({"prb_insts_12" : "sum", "perc_paym_inst" : "min"})
    application = application.join(inst_pay)
    del inst_pay
    gc.collect()
    
    # How many times client paid less than 90% the minimal amount of the instalment in the last 6 months ?
    inst_pay = installments_payments.copy()
    inst_pay["prb_insts_6"] = 0
    inst_pay.loc[((inst_pay["AMT_PAYMENT"]/inst_pay["AMT_INSTALMENT"]) < 0.9) & (inst_pay["DAYS_INSTALMENT"] > -183), "prb_insts_6"] = 1
    inst_pay = inst_pay.groupby("SK_ID_CURR").agg({"prb_insts_6" : "sum"})
    application = application.join(inst_pay)
    del inst_pay
    gc.collect()
    
    # How many times client paid less than 90% the minimal amount of the instalment in the last 3 months ?
    inst_pay = installments_payments.copy()
    inst_pay["prb_insts_3"] = 0
    inst_pay.loc[((inst_pay["AMT_PAYMENT"]/inst_pay["AMT_INSTALMENT"]) < 0.9) & (inst_pay["DAYS_INSTALMENT"] > -92), "prb_insts_3"] = 1
    inst_pay = inst_pay.groupby("SK_ID_CURR").agg({"prb_insts_3" : "sum"})
    application = application.join(inst_pay)
    del inst_pay
    gc.collect()
    
    # Prepare train and test dataframes
    
    installments_payments = installments_payments.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending = True)
    previous_application = previous_application.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending = True)
    bureau = bureau.sort_values(["SK_ID_CURR", "DAYS_CREDIT"], ascending = True)
    
    print("prepare installments payments")
    #installments_payments all time
    installments_payments["PAYMENT_DIFF"] = installments_payments["AMT_PAYMENT"] - installments_payments["AMT_INSTALMENT"]
    installments_payments["DBD"] = installments_payments["DAYS_INSTALMENT"] - installments_payments["DAYS_ENTRY_PAYMENT"]
    installments_payments["DPD"] = installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]
    installments_payments['DPD'] = installments_payments['DPD'].apply(lambda x: x if x > 0 else 0)
    installments_payments['DBD'] = installments_payments['DBD'].apply(lambda x: x if x > 0 else 0)
    installments_payments['AMT_INSTALMENT_div_AMT_PAYMENT'] = installments_payments['AMT_INSTALMENT'].divide(installments_payments['AMT_PAYMENT'].replace(0, np.nan))
    installments_payments['AMT_PAYMENT_div_DAYS_ENTRY_PAYMENT'] = installments_payments['AMT_PAYMENT'].divide(installments_payments['DAYS_ENTRY_PAYMENT'].replace(0, np.nan))
    installments_payments['DAYS_INSTALMENT_div_DAYS_ENTRY_PAYMENT'] = installments_payments['DAYS_INSTALMENT'].divide(installments_payments['DAYS_ENTRY_PAYMENT'])
    inst_pay_aggs = {"DAYS_ENTRY_PAYMENT" : ["max"],
                   "DAYS_INSTALMENT" : ["max"]}
    inst_pay = installments_payments.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(inst_pay)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    inst_pay_aggs = {"DBD" : ["max"], "DPD" : ["max"],
                   "PAYMENT_DIFF" : ["min", "max", "mean"], 
                     "AMT_PAYMENT" : ["max", "mean", "min", "sum"],
                    "AMT_INSTALMENT" : ["max", "mean", "min", "sum"]}
    inst_pay = installments_payments.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(inst_pay)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    inst_pay_aggs = {"AMT_PAYMENT_div_DAYS_ENTRY_PAYMENT" : ["max"],
                   "DAYS_INSTALMENT_div_DAYS_ENTRY_PAYMENT" : ["max"], 
                   "NUM_INSTALMENT_VERSION" : ["mean"],
                    "PAYMENT_DIFF" : ["last"],
                    "DBD" : ["last"],
                    "DPD" : ["last"]}
    inst_pay = installments_payments.sort_values(['SK_ID_CURR', 'DAYS_ENTRY_PAYMENT'])
    inst_pay = inst_pay.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    application = application.join(inst_pay)
    application = application.join(installments_payments.groupby("SK_ID_PREV").agg({"SK_ID_CURR": "min", "AMT_PAYMENT" : "min"}).groupby("SK_ID_CURR").agg({"AMT_PAYMENT" : "sum"})
                       , rsuffix = "_min_sum_INST")
    
    #installments_payments 6 months
    inst_pay_6 = installments_payments.loc[installments_payments["DAYS_INSTALMENT"] > -182]
    application = application.join(inst_pay_6.groupby("SK_ID_PREV").agg({"SK_ID_CURR": "min", "AMT_PAYMENT" : "min"}).groupby("SK_ID_CURR").agg({"AMT_PAYMENT" : "sum"})
                       , rsuffix = "_min_sum_6_INST")
    inst_pay_aggs = {"DBD" : ["max"], 
                     "DPD" : ["max"], 
                     "PAYMENT_DIFF" : ["mean"]}
    inst_pay = inst_pay_6.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_6_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(inst_pay)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    inst_pay_aggs = {"DAYS_ENTRY_PAYMENT" : ["max"], 
                     "DAYS_INSTALMENT" : ["max"]}
    inst_pay = inst_pay_6.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_6_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(inst_pay)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    inst_pay_aggs = {"NUM_INSTALMENT_VERSION" : ["mean"]}
    inst_pay = inst_pay_6.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_6_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    application = application.join(inst_pay)
    
    #installments_payments 12 months
    inst_pay_12 = installments_payments.loc[installments_payments["DAYS_INSTALMENT"] > -366]
    application = application.join(inst_pay_12.groupby("SK_ID_PREV").agg({"SK_ID_CURR": "min", "AMT_PAYMENT" : "min"}).groupby("SK_ID_CURR").agg({"AMT_PAYMENT" : "sum"})
                       , rsuffix = "_min_sum_12_INST")
    inst_pay_aggs = {"DBD" : ["max"], 
                     "DPD" : ["max"], 
                     "PAYMENT_DIFF" : ["mean"]}
    inst_pay = inst_pay_12.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_12_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(inst_pay)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    inst_pay_aggs = {"DAYS_ENTRY_PAYMENT" : ["max"], 
                     "DAYS_INSTALMENT" : ["max"]}
    inst_pay = inst_pay_12.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_12_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(inst_pay)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    inst_pay_aggs = {"NUM_INSTALMENT_VERSION" : ["mean"]}
    inst_pay = inst_pay_12.groupby("SK_ID_CURR").agg(inst_pay_aggs)
    inst_pay.columns = pd.Index(['INST_PAY_12_' + e[0] + "_" + e[1].upper() for e in inst_pay.columns.tolist()])
    application = application.join(inst_pay)
    
    #POS_CASH_balance all time
    print("prepare POS CASH balance")
    application = application.join(POS_CASH_balance.groupby("SK_ID_PREV").agg({"SK_ID_CURR": "min", "CNT_INSTALMENT_FUTURE" : "mean"}).groupby("SK_ID_CURR").agg({"CNT_INSTALMENT_FUTURE" : "max"})
                       , rsuffix = "_mean_max_POS")
    pos_cash_aggs = {"CNT_INSTALMENT_FUTURE" : ["mean"], "MONTHS_BALANCE" : ["max"],
                      "SK_DPD" : ["max"], "SK_DPD_DEF" : ["max"]}
    pos_cash_bal = POS_CASH_balance.groupby("SK_ID_CURR").agg(pos_cash_aggs)
    pos_cash_bal.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_cash_bal.columns.tolist()])
    application = application.join(pos_cash_bal)
    application = application.join(POS_CASH_balance.groupby("SK_ID_CURR").NAME_CONTRACT_STATUS.value_counts().unstack().fillna(0), rsuffix = "_POS")
    
    #POS_CASH_balance 6 months
    POS_CASH_balance_6 = POS_CASH_balance.loc[POS_CASH_balance["MONTHS_BALANCE"] >= -6]
    application = application.join(POS_CASH_balance_6.groupby("SK_ID_PREV").agg({"SK_ID_CURR": "min", "CNT_INSTALMENT_FUTURE" : "mean"}).groupby("SK_ID_CURR").agg({"CNT_INSTALMENT_FUTURE" : "max"})
                       , rsuffix = "_mean_max_6_POS")
    pos_cash_bal = POS_CASH_balance_6.groupby("SK_ID_CURR").agg(pos_cash_aggs)
    pos_cash_bal.columns = pd.Index(['POS_6_' + e[0] + "_" + e[1].upper() for e in pos_cash_bal.columns.tolist()])
    application = application.join(pos_cash_bal)
    application = application.join(POS_CASH_balance_6.groupby("SK_ID_CURR").NAME_CONTRACT_STATUS.value_counts().unstack().fillna(0), rsuffix = "_6_POS")

    #POS_CASH_balance 12 months
    POS_CASH_balance_12 = POS_CASH_balance.loc[POS_CASH_balance["MONTHS_BALANCE"] >= -12]
    application = application.join(POS_CASH_balance_12.groupby("SK_ID_PREV").agg({"SK_ID_CURR": "min", "CNT_INSTALMENT_FUTURE" : "mean"}).groupby("SK_ID_CURR").agg({"CNT_INSTALMENT_FUTURE" : "max"})
                       , rsuffix = "_mean_max_12_POS")
    pos_cash_bal = POS_CASH_balance_12.groupby("SK_ID_CURR").agg(pos_cash_aggs)
    pos_cash_bal.columns = pd.Index(['POS_12_' + e[0] + "_" + e[1].upper() for e in pos_cash_bal.columns.tolist()])
    application = application.join(pos_cash_bal)
    application = application.join(POS_CASH_balance_12.groupby("SK_ID_CURR").NAME_CONTRACT_STATUS.value_counts().unstack().fillna(0), rsuffix = "_12_POS")
    
    #credit_card_balance all time
    print("prepare credit card balance")
    credit_card_balance["AMT_DRAWINGS"] = credit_card_balance["AMT_DRAWINGS_ATM_CURRENT"] + credit_card_balance["AMT_DRAWINGS_CURRENT"] + credit_card_balance["AMT_DRAWINGS_OTHER_CURRENT"] + credit_card_balance["AMT_DRAWINGS_POS_CURRENT"]
    credit_card_balance["CNT_DRAWINGS"] = credit_card_balance["CNT_DRAWINGS_ATM_CURRENT"] + credit_card_balance["CNT_DRAWINGS_CURRENT"] + credit_card_balance["CNT_DRAWINGS_OTHER_CURRENT"] + credit_card_balance["CNT_DRAWINGS_POS_CURRENT"]
    credit_card_balance["AMT_CREDIT_OVERDUE"] = credit_card_balance["AMT_BALANCE"] - credit_card_balance["AMT_PAYMENT_TOTAL_CURRENT"]
    credit_card_balance["AMT_CREDIT"] = credit_card_balance["SK_ID_PREV"].map(previous_application.set_index("SK_ID_PREV")["AMT_CREDIT"])
    credit_card_aggs = {
                        "CNT_DRAWINGS_ATM_CURRENT" : ["mean"], "CNT_DRAWINGS" : ["mean"],
                          "AMT_BALANCE" : ["max"], "AMT_DRAWINGS" : ["mean"],
                          "SK_DPD" : ["max"], "SK_DPD_DEF" : ["max"],
                          "MONTHS_BALANCE" : ["max"]}
    cred_card_bal = credit_card_balance.groupby("SK_ID_CURR").agg(credit_card_aggs)    
    cred_card_bal.columns = pd.Index(['CRE_' + e[0] + "_" + e[1].upper() for e in cred_card_bal.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(cred_card_bal)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    new_columns = [col for col in new_columns if "SK_DPD" not in col]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    new_columns = [col for col in new_columns if "SK_DPD" not in col]
    credit_card_aggs_1 = {"AMT_PAYMENT_TOTAL_CURRENT" : "sum", 
                          "AMT_BALANCE" : "sum", "AMT_DRAWINGS" : "sum", 
                          "AMT_CREDIT_LIMIT_ACTUAL" : "sum",
                          "AMT_INST_MIN_REGULARITY" : "sum",
                          "AMT_CREDIT_OVERDUE" : "sum",
                          "AMT_CREDIT" : "sum"}
    credit_card_aggs_2 = {"BALANCE_TO_LIMIT_RATIO" : ["mean", "max"],
                         "DRAWINGS_TO_LIMIT_RATIO" : ["mean", "max"],
#                          "OVERDUE_TO_AMT_CREDIT_RATIO" : ["mean", "max"],
                         "AMT_PAYMENT_TOTAL_CURRENT_TO_MIN_INST_RATIO" : ["min", "mean", "max"]}
    cred_card_bal = credit_card_balance.groupby(["SK_ID_CURR", "MONTHS_BALANCE"]).agg(credit_card_aggs_1)
    cred_card_bal["BALANCE_TO_LIMIT_RATIO"] = cred_card_bal["AMT_BALANCE"].divide((cred_card_bal["AMT_CREDIT_LIMIT_ACTUAL"]))
    cred_card_bal["DRAWINGS_TO_LIMIT_RATIO"] = cred_card_bal["AMT_DRAWINGS"].divide((cred_card_bal["AMT_CREDIT_LIMIT_ACTUAL"]))
    cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT_TO_LIMIT_RATIO"] = cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT"].divide((cred_card_bal["AMT_CREDIT_LIMIT_ACTUAL"]))
    cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT_TO_MIN_INST_RATIO"] = cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT"].divide((cred_card_bal["AMT_INST_MIN_REGULARITY"]))
    cred_card_bal["OVERDUE_TO_AMT_CREDIT_RATIO"] = cred_card_bal["AMT_CREDIT_OVERDUE"].divide(cred_card_bal["AMT_CREDIT"].replace(0,np.nan))
    cred_card_bal = cred_card_bal.reset_index().groupby("SK_ID_CURR").agg(credit_card_aggs_2)
    cred_card_bal.columns = pd.Index(['CRE_' + e[0] + "_" + e[1].upper() for e in cred_card_bal.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(cred_card_bal)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 

    #credit_card_balance 6 months
    credit_card_balance_6 = credit_card_balance.loc[credit_card_balance["MONTHS_BALANCE"] >= -6]
    del credit_card_aggs["MONTHS_BALANCE"]
    cred_card_bal = credit_card_balance_6.groupby("SK_ID_CURR").agg(credit_card_aggs)    
    cred_card_bal.columns = pd.Index(['CRE_6_' + e[0] + "_" + e[1].upper() for e in cred_card_bal.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(cred_card_bal)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    new_columns = [col for col in new_columns if "SK_DPD" not in col]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    new_columns = [col for col in new_columns if "SK_DPD" not in col]
    cred_card_bal = credit_card_balance_6.groupby(["SK_ID_CURR", "MONTHS_BALANCE"]).agg(credit_card_aggs_1)
    cred_card_bal["BALANCE_TO_LIMIT_RATIO"] = cred_card_bal["AMT_BALANCE"].divide((cred_card_bal["AMT_CREDIT_LIMIT_ACTUAL"]))
    cred_card_bal["DRAWINGS_TO_LIMIT_RATIO"] = cred_card_bal["AMT_DRAWINGS"].divide((cred_card_bal["AMT_CREDIT_LIMIT_ACTUAL"]))
    cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT_TO_LIMIT_RATIO"] = cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT"].divide((cred_card_bal["AMT_CREDIT_LIMIT_ACTUAL"]))
    cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT_TO_MIN_INST_RATIO"] = cred_card_bal["AMT_PAYMENT_TOTAL_CURRENT"].divide((cred_card_bal["AMT_INST_MIN_REGULARITY"]))
    cred_card_bal["UTILIZATION_RATE"] = cred_card_bal["AMT_BALANCE"].divide(cred_card_bal["AMT_CREDIT"].replace(0,np.nan))
    cred_card_bal["OVERDUE_TO_AMT_CREDIT_RATIO"] = cred_card_bal["AMT_CREDIT_OVERDUE"].divide(cred_card_bal["AMT_CREDIT"].replace(0,np.nan))
    credit_card_aggs_2["UTILIZATION_RATE"] = ["mean", "max"]
    credit_card_aggs_2["OVERDUE_TO_AMT_CREDIT_RATIO"] = ["mean", "max"]
    cred_card_bal = cred_card_bal.reset_index().groupby("SK_ID_CURR").agg(credit_card_aggs_2)
    cred_card_bal.columns = pd.Index(['CRE_6_' + e[0] + "_" + e[1].upper() for e in cred_card_bal.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(cred_card_bal)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(np.inf) 

    #credit_card_balance 12 months
    credit_card_balance_12 = credit_card_balance.loc[credit_card_balance["MONTHS_BALANCE"] >= -12]
    cred_card_bal = credit_card_balance_12.groupby("SK_ID_CURR").agg(credit_card_aggs)    
    cred_card_bal.columns = pd.Index(['CRE_12_' + e[0] + "_" + e[1].upper() for e in cred_card_bal.columns.tolist()])
    train_columns_before = application.columns.tolist()
    application = application.join(cred_card_bal)
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    new_columns = [col for col in new_columns if "SK_DPD" not in col]
    application[new_columns] = application[new_columns].fillna(np.inf) 
    new_columns = [col for col in new_columns if "SK_DPD" not in col]

    #bureau_balance all time
    print("prepare bureau balance")
    bur_bal = bureau_balance.copy()
    application = application.join(bur_bal.groupby("SK_ID_CURR").STATUS.value_counts().unstack().fillna(0), rsuffix = "_BUR_BAL")
    application = application.join(bur_bal.groupby("SK_ID_CURR").agg({"MONTHS_BALANCE" : "max"}), rsuffix = "_max_BUR_BAL")
    
    #bureau_balance 6 months
    bur_bal = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] >= -6]
    application = application.join(bur_bal.groupby("SK_ID_CURR").STATUS.value_counts().unstack().fillna(0), rsuffix = "_6_BUR_BAL")
    
    #bureau_balance 12 months
    bur_bal = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] >= -12]
    application = application.join(bur_bal.groupby("SK_ID_CURR").STATUS.value_counts().unstack().fillna(0), rsuffix = "_12_BUR_BAL")

    #previous_application all time
    print("prepare previous application")
    previous_application["DAYS_LAST_DUE_div_DAYS_FIRST_DUE"] = previous_application["DAYS_LAST_DUE"].divide(previous_application["DAYS_FIRST_DUE"].replace(0, np.nan))
    cat_prev_app = previous_application.groupby("SK_ID_CURR").NAME_CONTRACT_STATUS.value_counts().unstack().fillna(0)
    application = application.join(cat_prev_app, rsuffix = "_PREV")
    cat_prev_app = previous_application.groupby("SK_ID_CURR").NAME_YIELD_GROUP.value_counts().unstack().fillna(0)
    application = application.join(cat_prev_app, rsuffix = "_PREV")
    prev_app_aggs = {"CNT_PAYMENT" : ["mean"], "AMT_DOWN_PAYMENT" : ["sum"],
                      "RATE_DOWN_PAYMENT" : ["max"], "HOUR_APPR_PROCESS_START" : ["mean"],
                      "DAYS_DECISION" : ["max"], "SELLERPLACE_AREA" : ["mean"],
                      "AMT_ANNUITY" : ["mean"], "AMT_APPLICATION" : ["mean"],
                      "DAYS_LAST_DUE_div_DAYS_FIRST_DUE" : ["max", "min"]}
    prev_app = previous_application.groupby("SK_ID_CURR").agg(prev_app_aggs)
    prev_app_aggs["DAYS_DECISION"] = ["max"]
    prev_app.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_app.columns.tolist()])
    application = application.join(prev_app, rsuffix = "_PREV")
    
    #previous_application accepted
    accepted = previous_application.loc[previous_application["NAME_CONTRACT_STATUS"] == "Approved"]
    cat_prev_app = accepted.groupby("SK_ID_CURR").NAME_YIELD_GROUP.value_counts().unstack().fillna(0)
    application = application.join(cat_prev_app, rsuffix = "_APP_PREV")
    prev_app = accepted.groupby("SK_ID_CURR").agg(prev_app_aggs)
    prev_app.columns = pd.Index(['PREV_APP_' + e[0] + "_" + e[1].upper() for e in prev_app.columns.tolist()])
    application = application.join(prev_app, rsuffix = "_APP_PREV")
    
    #previous_application refused
    refused = previous_application.loc[previous_application["NAME_CONTRACT_STATUS"] == "Refused"]
    cat_prev_app = refused.groupby("SK_ID_CURR").NAME_YIELD_GROUP.value_counts().unstack().fillna(0)
    application = application.join(cat_prev_app, rsuffix = "_REF_PREV")
    del prev_app_aggs["DAYS_LAST_DUE_div_DAYS_FIRST_DUE"]
    prev_app = refused.groupby("SK_ID_CURR").agg(prev_app_aggs)
    prev_app.columns = pd.Index(['PREV_APP_' + e[0] + "_" + e[1].upper() for e in prev_app.columns.tolist()])
    application = application.join(prev_app, rsuffix = "_REF_PREV")
    
    #previous_application 6 months
    ids_1 = POS_CASH_balance.loc[POS_CASH_balance["MONTHS_BALANCE"] >= -6]["SK_ID_PREV"].unique()
    ids_2 = credit_card_balance.loc[credit_card_balance["MONTHS_BALANCE"] >= -6]["SK_ID_PREV"].unique()
    ids = np.unique(np.append(ids_1, ids_2))
    active = previous_application.loc[previous_application["SK_ID_PREV"].isin(ids)]
    cat_prev_app = active.groupby("SK_ID_CURR").NAME_YIELD_GROUP.value_counts().unstack().fillna(0)
    application = application.join(cat_prev_app, rsuffix = "_6_PREV")
    prev_app_aggs["DAYS_LAST_DUE_div_DAYS_FIRST_DUE"] = ["max", "min"]
    prev_app = active.groupby("SK_ID_CURR").agg(prev_app_aggs)
    prev_app.columns = pd.Index(['PREV_6_' + e[0] + "_" + e[1].upper() for e in prev_app.columns.tolist()])
    application = application.join(prev_app, rsuffix = "_6_PREV")
    
    #previous_application 12 months
    ids_1 = POS_CASH_balance.loc[POS_CASH_balance["MONTHS_BALANCE"] >= -12]["SK_ID_PREV"].unique()
    ids_2 = credit_card_balance.loc[credit_card_balance["MONTHS_BALANCE"] >= -12]["SK_ID_PREV"].unique()
    ids = np.unique(np.append(ids_1, ids_2))
    active = previous_application.loc[previous_application["SK_ID_PREV"].isin(ids)]
    cat_prev_app = active.groupby("SK_ID_CURR").NAME_YIELD_GROUP.value_counts().unstack().fillna(0)
    application = application.join(cat_prev_app, rsuffix = "_12_PREV")
    prev_app = active.groupby("SK_ID_CURR").agg(prev_app_aggs)
    prev_app.columns = pd.Index(['PREV_12_' + e[0] + "_" + e[1].upper() for e in prev_app.columns.tolist()])
    application = application.join(prev_app, rsuffix = "_12_PREV")
    
    
    #bureau
    print("prepare bureau")
    bureau["AMT_CREDIT_SUM_div_AMT_CREDIT_SUM_DEBT"] = bureau["AMT_CREDIT_SUM"].divide(bureau["AMT_CREDIT_SUM_DEBT"].replace(0, np.nan))
    bureau["AMT_CREDIT_SUM_diff_AMT_CREDIT_SUM_DEBT"] = bureau["AMT_CREDIT_SUM"].subtract(bureau["AMT_CREDIT_SUM_DEBT"])
    bureau["DAYS_CREDIT_div_AMT_CREDIT_SUM"] = bureau["DAYS_CREDIT"].divide(bureau["AMT_CREDIT_SUM"].replace(0, np.nan))
    bureau["DAYS_CREDIT_diff_DAYS_CREDIT_ENDDATE"] = bureau["DAYS_CREDIT"].subtract(bureau["DAYS_CREDIT_ENDDATE"])
    bureau["DAYS_CREDIT_div_DAYS_CREDIT_ENDDATE"] = bureau["DAYS_CREDIT"].divide(bureau["DAYS_CREDIT_ENDDATE"].replace(0, np.nan))
    bureau["AMT_CREDIT_SUM_div_DAYS_ENDDATE_FACT"] = bureau["AMT_CREDIT_SUM"].divide(bureau["DAYS_ENDDATE_FACT"].replace(0, np.nan))
    bureau["DAYS_CREDIT_diff_DAYS_ENDDATE_FACT"] = bureau["DAYS_CREDIT"].subtract(bureau["DAYS_ENDDATE_FACT"])
    bureau["AMT_CREDIT_MAX_OVERDUE_div_DAYS_CREDIT"] = bureau["AMT_CREDIT_MAX_OVERDUE"].divide(bureau["DAYS_CREDIT"].replace(0, np.nan))
    bureau["DAYS_ENDDATE_FACT_div_DAYS_CREDIT_ENDDATE"] = bureau["DAYS_ENDDATE_FACT"].divide(bureau["DAYS_CREDIT_ENDDATE"].replace(0, np.nan))
    bur_aggs = {"DAYS_ENDDATE_FACT_div_DAYS_CREDIT_ENDDATE" : ["max", "min"],
               "DAYS_CREDIT" : ["max"], "DAYS_CREDIT_ENDDATE" : ["max"],
               "DAYS_ENDDATE_FACT" : ["max"], "DAYS_CREDIT_UPDATE" : ["max"],
               "DAYS_CREDIT_div_AMT_CREDIT_SUM" : ["max"], 
               "DAYS_CREDIT_diff_DAYS_CREDIT_ENDDATE" : ["max"],
               "AMT_CREDIT_SUM_div_AMT_CREDIT_SUM_DEBT" : ["min"],
               "DAYS_CREDIT_div_DAYS_CREDIT_ENDDATE" : ["min"],
               "AMT_CREDIT_SUM_diff_AMT_CREDIT_SUM_DEBT" : ["min"],
               "AMT_CREDIT_SUM_div_DAYS_ENDDATE_FACT" : ["min"],
               "AMT_CREDIT_MAX_OVERDUE_div_DAYS_CREDIT" : ["min"],
               "DAYS_CREDIT_diff_DAYS_ENDDATE_FACT" : ["max"]
               }
    bur = bureau.groupby("SK_ID_CURR").agg(bur_aggs)   
    bur.columns = pd.Index(['BUR_' + e[0] + "_" + e[1].upper() for e in bur.columns.tolist()])
    application = application.join(bur)
    bur_aggs_1 = {"AMT_CREDIT_SUM_DEBT" : ["mean", "sum"],
                    "AMT_CREDIT_SUM" : ["max"], 
                  "AMT_CREDIT_MAX_OVERDUE" : ["mean", "max"]}
    bur = bureau.groupby("SK_ID_CURR").agg(bur_aggs_1)   
    bur.columns = pd.Index(['BUR_' + e[0] + "_" + e[1].upper() for e in bur.columns.tolist()])
    application = application.join(bur)
    bur = bureau.groupby("SK_ID_CURR").CREDIT_ACTIVE.value_counts().unstack().fillna(0)
    application = application.join(bur, rsuffix = "_BUR")
    bur = bureau.groupby("SK_ID_CURR").CREDIT_TYPE.value_counts().unstack().fillna(0)
    application = application.join(bur, rsuffix = "_BUR")
    
    #bureau 1 year data
    bureau_12 = bureau.loc[bureau["DAYS_CREDIT"] >= -365]
    bur = bureau_12.groupby("SK_ID_CURR").agg(bur_aggs)   
    bur.columns = pd.Index(['BUR_12_' + e[0] + "_" + e[1].upper() for e in bur.columns.tolist()])
    application = application.join(bur)
    train_columns_before = application.columns.tolist()
    bur = bureau_12.groupby("SK_ID_CURR").agg(bur_aggs_1)   
    bur.columns = pd.Index(['BUR_12_' + e[0] + "_" + e[1].upper() for e in bur.columns.tolist()])
    application = application.join(bur)
    bur = bureau_12.groupby("SK_ID_CURR").CREDIT_ACTIVE.value_counts().unstack().fillna(0)
    application = application.join(bur, rsuffix = "_12_BUR")
    bur = bureau_12.groupby("SK_ID_CURR").CREDIT_TYPE.value_counts().unstack().fillna(0)
    application = application.join(bur, rsuffix = "_12_BUR")
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(0) 
    
    #bureau 6 months data
    bureau_6 = bureau.loc[bureau["DAYS_CREDIT"] >= -183]
    bur = bureau_6.groupby("SK_ID_CURR").agg(bur_aggs)   
    bur.columns = pd.Index(['BUR_6_' + e[0] + "_" + e[1].upper() for e in bur.columns.tolist()])
    application = application.join(bur)
    train_columns_before = application.columns.tolist()
    bur = bureau_6.groupby("SK_ID_CURR").agg(bur_aggs_1)   
    bur.columns = pd.Index(['BUR_6_' + e[0] + "_" + e[1].upper() for e in bur.columns.tolist()])
    application = application.join(bur)
    bur = bureau_6.groupby("SK_ID_CURR").CREDIT_ACTIVE.value_counts().unstack().fillna(0)
    application = application.join(bur, rsuffix = "_6_BUR")
    bur = bureau_6.groupby("SK_ID_CURR").CREDIT_TYPE.value_counts().unstack().fillna(0)
    application = application.join(bur, rsuffix = "_6_BUR")
    new_columns = [col for col in application.columns.tolist() if col not in train_columns_before]
    application[new_columns] = application[new_columns].fillna(0) 
    
    #credit cards
    print("prepare credit cards")
    credit_card_balance["STATUS"] = (credit_card_balance["SK_DPD"]/30).astype(int)
    credit_card_balance["STATUS"] = np.where(credit_card_balance["STATUS"] >= 5, 5, credit_card_balance["STATUS"])
    credit_card_balance["STATUS"] = credit_card_balance["STATUS"].astype('category')
    bur_bal = bureau_balance[["SK_ID_BUREAU", "STATUS"]]
    bur_bal["STATUS"] = bur_bal["STATUS"].replace({"C" : 0, "X" : 0, "0" : 0, "1" : 1, "2": 2, "3" : 3, "4": 4, "5" : 5})
    bur_bal = bur_bal.groupby("SK_ID_BUREAU").STATUS.value_counts().unstack().fillna(0)
    bur_bal.columns = pd.Index(['STATUS_' + str(e) for e in bur_bal.columns.tolist()])    
    credit_cards = bureau.loc[(bureau["CREDIT_TYPE"] == "Credit card")][["SK_ID_BUREAU", "SK_ID_CURR", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT"]]
    credit_cards = credit_cards.set_index("SK_ID_BUREAU").join(bur_bal)
    credit_cards = credit_cards.reset_index().drop(columns = ["SK_ID_BUREAU"])
    credit_cards = credit_cards.rename(index = str, columns = {"AMT_CREDIT_SUM" : "AMT_CREDIT", 
                                                             "AMT_CREDIT_SUM_DEBT" : "AMT_TOTAL_RECEIVABLE", 
                                                            "AMT_CREDIT_SUM_LIMIT" : "AMT_CREDIT_LIMIT_ACTUAL"
                                                              })
    cred = credit_card_balance[["SK_ID_CURR", "SK_ID_PREV", "AMT_CREDIT", "AMT_TOTAL_RECEIVABLE", "AMT_CREDIT_LIMIT_ACTUAL"]].groupby("SK_ID_PREV").last()
    cred = cred.groupby("SK_ID_PREV").agg({"AMT_CREDIT" : "sum", "AMT_TOTAL_RECEIVABLE" : "sum", "AMT_CREDIT_LIMIT_ACTUAL" : "sum"})
    status = credit_card_balance.groupby("SK_ID_PREV").STATUS.value_counts().unstack().fillna(0)
    status.columns = pd.Index(['STATUS_' + str(e) for e in status.columns.tolist()])
    cred = cred.join(status)
    credit_cards = credit_cards.append(cred.reset_index().drop(columns = ["SK_ID_PREV"]))
    credit_cards = credit_cards.groupby("SK_ID_CURR").agg({"AMT_CREDIT" : ["sum"], 
                                                           "AMT_TOTAL_RECEIVABLE" : ["sum"],
                                                          "STATUS_0" : ["sum"],"STATUS_1" : ["sum"],
                                                          "STATUS_2" : ["sum"], "STATUS_3" : ["sum"],
                                                          "STATUS_4" : ["sum"],"STATUS_5" : ["sum"]})
    credit_cards.columns = pd.Index(['CRED_CAR_' + e[0] + "_" + e[1].upper() for e in credit_cards.columns.tolist()])
    credit_cards_columns = credit_cards.columns.tolist()
    credit_cards["CRED_CAR_AMT_TOTAL_RECEIVABLE_SUM_div_CRED_CAR_AMT_CREDIT_SUM"] = credit_cards["CRED_CAR_AMT_TOTAL_RECEIVABLE_SUM"].divide(credit_cards["CRED_CAR_AMT_CREDIT_SUM"].replace(0,np.nan))
    application = application.join(credit_cards)
    
    #credit cards 6 months
    bur_bal = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] >= -6][["SK_ID_BUREAU", "STATUS"]]
    bur_bal["STATUS"] = bur_bal["STATUS"].replace({"C" : 0, "X" : 0, "0" : 0, "1" : 1, "2": 2, "3" : 3, "4": 4, "5" : 5})
    bur_bal = bur_bal.groupby("SK_ID_BUREAU").STATUS.value_counts().unstack().fillna(0)
    bur_bal.columns = pd.Index(['STATUS_' + str(e) for e in bur_bal.columns.tolist()])    
    credit_cards =  bureau.loc[(bureau["CREDIT_TYPE"] == "Credit card") 
                              & ((bureau["DAYS_CREDIT_ENDDATE"] >= -183)
                             | (bureau["DAYS_ENDDATE_FACT"] >= -183))][["SK_ID_BUREAU", "SK_ID_CURR", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT"]]
    credit_cards = credit_cards.set_index("SK_ID_BUREAU").join(bur_bal)
    credit_cards = credit_cards.reset_index().drop(columns = ["SK_ID_BUREAU"])
    credit_cards = credit_cards.rename(index = str, columns = {"AMT_CREDIT_SUM" : "AMT_CREDIT", 
                                                             "AMT_CREDIT_SUM_DEBT" : "AMT_TOTAL_RECEIVABLE", 
                                                            "AMT_CREDIT_SUM_LIMIT" : "AMT_CREDIT_LIMIT_ACTUAL"
                                                              })
    cred = credit_card_balance[["SK_ID_CURR", "SK_ID_PREV", "AMT_CREDIT", "AMT_TOTAL_RECEIVABLE", "AMT_CREDIT_LIMIT_ACTUAL"]].groupby("SK_ID_PREV").last()
    cred = cred.groupby("SK_ID_PREV").agg({"AMT_CREDIT" : "sum", "AMT_TOTAL_RECEIVABLE" : "sum", "AMT_CREDIT_LIMIT_ACTUAL" : "sum"})
    status = credit_card_balance.groupby("SK_ID_PREV").STATUS.value_counts().unstack().fillna(0)
    status.columns = pd.Index(['STATUS_6_' + str(e) for e in status.columns.tolist()])
    cred = cred.join(status)
    credit_cards = credit_cards.append(cred.reset_index().drop(columns = ["SK_ID_PREV"]))
    credit_cards = credit_cards.groupby("SK_ID_CURR").agg({"AMT_CREDIT" : ["sum"], 
                                                           "AMT_TOTAL_RECEIVABLE" : ["sum"],
                                                          "STATUS_0" : ["sum"],"STATUS_1" : ["sum"],
                                                          "STATUS_2" : ["sum"], "STATUS_3" : ["sum"],
                                                          "STATUS_4" : ["sum"],"STATUS_5" : ["sum"]})
    credit_cards.columns = pd.Index(['CRED_CAR_6_' + e[0] + "_" + e[1].upper() for e in credit_cards.columns.tolist()])
    credit_cards_columns = credit_cards.columns.tolist()
    credit_cards["CRED_CAR_6_AMT_TOTAL_RECEIVABLE_SUM_div_CRED_CAR_AMT_CREDIT_SUM"] = credit_cards["CRED_CAR_6_AMT_TOTAL_RECEIVABLE_SUM"].divide(credit_cards["CRED_CAR_6_AMT_CREDIT_SUM"].replace(0,np.nan))
    application = application.join(credit_cards)
    
    #credit cards 12 months
    bur_bal = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] >= -12][["SK_ID_BUREAU", "STATUS"]]
    bur_bal["STATUS"] = bur_bal["STATUS"].replace({"C" : 0, "X" : 0, "0" : 0, "1" : 1, "2": 2, "3" : 3, "4": 4, "5" : 5})
    bur_bal = bur_bal.groupby("SK_ID_BUREAU").STATUS.value_counts().unstack().fillna(0)
    bur_bal.columns = pd.Index(['STATUS_' + str(e) for e in bur_bal.columns.tolist()])    
    credit_cards =  bureau.loc[(bureau["CREDIT_TYPE"] == "Credit card") 
                              & ((bureau["DAYS_CREDIT_ENDDATE"] >= -365)
                             | (bureau["DAYS_ENDDATE_FACT"] >= -365))][["SK_ID_BUREAU", "SK_ID_CURR", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT"]]
    credit_cards = credit_cards.set_index("SK_ID_BUREAU").join(bur_bal)
    credit_cards = credit_cards.reset_index().drop(columns = ["SK_ID_BUREAU"])
    credit_cards = credit_cards.rename(index = str, columns = {"AMT_CREDIT_SUM" : "AMT_CREDIT", 
                                                             "AMT_CREDIT_SUM_DEBT" : "AMT_TOTAL_RECEIVABLE", 
                                                            "AMT_CREDIT_SUM_LIMIT" : "AMT_CREDIT_LIMIT_ACTUAL"
                                                              })
    cred = credit_card_balance[["SK_ID_CURR", "SK_ID_PREV", "AMT_CREDIT", "AMT_TOTAL_RECEIVABLE", "AMT_CREDIT_LIMIT_ACTUAL"]].groupby("SK_ID_PREV").last()
    cred = cred.groupby("SK_ID_PREV").agg({"AMT_CREDIT" : "sum", "AMT_TOTAL_RECEIVABLE" : "sum", "AMT_CREDIT_LIMIT_ACTUAL" : "sum"})
    status = credit_card_balance.groupby("SK_ID_PREV").STATUS.value_counts().unstack().fillna(0)
    status.columns = pd.Index(['STATUS_12_' + str(e) for e in status.columns.tolist()])
    cred = cred.join(status)
    credit_cards = credit_cards.append(cred.reset_index().drop(columns = ["SK_ID_PREV"]))
    credit_cards = credit_cards.groupby("SK_ID_CURR").agg({"AMT_CREDIT" : ["sum"], 
                                                           "AMT_TOTAL_RECEIVABLE" : ["sum"],
                                                          "STATUS_0" : ["sum"],"STATUS_1" : ["sum"],
                                                          "STATUS_2" : ["sum"], "STATUS_3" : ["sum"],
                                                          "STATUS_4" : ["sum"],"STATUS_5" : ["sum"]})
    credit_cards.columns = pd.Index(['CRED_CAR_12_' + e[0] + "_" + e[1].upper() for e in credit_cards.columns.tolist()])
    credit_cards_columns = credit_cards.columns.tolist()
    credit_cards["CRED_CAR_12_AMT_TOTAL_RECEIVABLE_SUM_div_CRED_CAR_AMT_CREDIT_SUM"] = credit_cards["CRED_CAR_12_AMT_TOTAL_RECEIVABLE_SUM"].divide(credit_cards["CRED_CAR_12_AMT_CREDIT_SUM"].replace(0,np.nan))
#     credit_cards = credit_cards.drop(columns = credit_cards_columns)
    application = application.join(credit_cards)
    
    # application
    print("prepare application")
    application["SK_DPD_TOT"] = application["POS_SK_DPD_MAX"] + application["CRE_SK_DPD_MAX"]
    application["SK_DPD_TOT_6"] = application["POS_6_SK_DPD_MAX"] + application["CRE_6_SK_DPD_MAX"]
    application["SK_DPD_TOT_12"] = application["POS_12_SK_DPD_MAX"] + application["CRE_12_SK_DPD_MAX"]
    application["SK_DPD_DEF_TOT"] = application["POS_SK_DPD_DEF_MAX"] + application["CRE_SK_DPD_DEF_MAX"]
    application["SK_DPD_DEF_TOT_6"] = application["POS_6_SK_DPD_DEF_MAX"] + application["CRE_6_SK_DPD_DEF_MAX"]
    application["SK_DPD_DEF_TOT_12"] = application["POS_12_SK_DPD_DEF_MAX"] + application["CRE_12_SK_DPD_DEF_MAX"]
    application["DAYS_EMPLOYED_div_DAYS_BIRTH"] = application["DAYS_EMPLOYED"].divide(application["DAYS_BIRTH"].replace(0, np.nan))
    application["AMT_INCOME_TOTAL_div_AMT_ANNUITY"] = application["AMT_INCOME_TOTAL"].divide(application["AMT_ANNUITY"].replace(0, np.nan))
    application["AMT_INCOME_TOTAL_div_CNT_FAM_MEMBERS"] = application["AMT_INCOME_TOTAL"].divide(application["CNT_FAM_MEMBERS"].replace(0, np.nan))
    application["REGION_RATING_CLIENT_W_CITY_div_EXT_SOURCE_3"] = application["REGION_RATING_CLIENT_W_CITY"].divide(application["EXT_SOURCE_3"])
    application["AMT_CREDIT_div_AMT_ANNUITY"] = application["AMT_CREDIT"].divide(application["AMT_ANNUITY"].replace(0, np.nan))
    application["DAYS_BIRTH_div_EXT_SOURCE_1"] = application["DAYS_BIRTH"].divide(application["EXT_SOURCE_1"].replace(0, np.nan))
    application["DAYS_BIRTH_div_EXT_SOURCE_2"] = application["DAYS_BIRTH"].divide(application["EXT_SOURCE_2"].replace(0, np.nan))
    application["DAYS_BIRTH_div_EXT_SOURCE_3"] = application["DAYS_BIRTH"].divide(application["EXT_SOURCE_3"].replace(0, np.nan))
    application["REGION_RATING_CLIENT_div_EXT_SOURCE_3"] = application["REGION_RATING_CLIENT"].divide(application["EXT_SOURCE_3"].replace(0, np.nan))
    application["AMT_CREDIT_div_AMT_GOODS_PRICE"] = application["AMT_CREDIT"].divide(application["AMT_GOODS_PRICE"].replace(0, np.nan))
    application["AMT_ANNUITY_div_EXT_SOURCE_2"] = application["AMT_ANNUITY"].divide(application["EXT_SOURCE_2"].replace(0, np.nan))
    application["AMT_ANNUITY_div_EXT_SOURCE_3"] = application["AMT_ANNUITY"].divide(application["EXT_SOURCE_3"].replace(0, np.nan))
    application["AMT_ANNUITY_div_EXT_SOURCE_1"] = application["AMT_ANNUITY"].divide(application["EXT_SOURCE_1"].replace(0, np.nan))
    application["CNT_FAM_MEMBERS_div_EXT_SOURCE_3"] = application["CNT_FAM_MEMBERS"].divide(application["EXT_SOURCE_3"].replace(0, np.nan))
    application["CNT_FAM_MEMBERS_div_EXT_SOURCE_3"] = application["CNT_FAM_MEMBERS"].divide(application["EXT_SOURCE_3"].replace(0, np.nan))
    application["DAYS_ID_PUBLISH_div_DAYS_LAST_PHONE_CHANGE"] = application["DAYS_ID_PUBLISH"].divide(application["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan))
    application["AMT_ANNUITY_div_DAYS_BIRTH"] = application["AMT_ANNUITY"].divide(application["DAYS_BIRTH"].replace(0, np.nan))
    application["AMT_INCOME_TOTAL_div_AMT_GOODS_PRICE"] = application["AMT_INCOME_TOTAL"].divide(application["AMT_GOODS_PRICE"].replace(0, np.nan))
    application["AMT_CREDIT_div_DAYS_EMPLOYED"] = application["AMT_CREDIT"].divide(application["DAYS_EMPLOYED"].replace(0, np.nan))
    application["AMT_INCOME_TOTAL_div_AMT_CREDIT"] = application["AMT_INCOME_TOTAL"].divide(application["AMT_CREDIT"].replace(0, np.nan))
    application["AMT_INCOME_TOTAL_div_REGION_POPULATION_RELATIVE"] = application["AMT_INCOME_TOTAL"].divide(application["REGION_POPULATION_RELATIVE"].replace(0, np.nan))
    application["AMT_ANNUITY_div_DAYS_EMPLOYED"] = application["AMT_ANNUITY"].divide(application["DAYS_EMPLOYED"].replace(0, np.nan))
    application["AMT_ANNUITY_div_AMT_GOODS_PRICE"] = application["AMT_ANNUITY"].divide(application["AMT_GOODS_PRICE"].replace(0, np.nan))
    application["AMT_CREDIT_diff_AMT_GOODS_PRICE"] = application["AMT_CREDIT"] - application["AMT_GOODS_PRICE"]
    application["last_credit"] = application[["PREV_DAYS_DECISION_MAX", "BUR_DAYS_CREDIT_MAX"]].max(axis = 1)
    application["curr_prb_bur_active"] = application["curr_prb_bur_active"].fillna(0)
    application["prbs_six_months_bur"] = application["prbs_six_months_bur"].fillna(0)
    application["prbs_twelve_months_bur"] = application["prbs_twelve_months_bur"].fillna(0)
    application["prbs_one_months_bur"] = application["prbs_one_months_bur"].fillna(0)
    application["perc_min_bur_diff_amt_crd"] = application["perc_min_bur_diff_amt_crd"].fillna(0)
    application["difficulties_past"] = application["difficulties_past"].fillna(0)
    application["perc_curr_bur_credit"] = application["perc_curr_bur_credit"].fillna(0)
    application["prb_and_lt_curr"] = application["prb_and_lt_curr"].fillna(0)
    application["amt_bur_credit_diff"] = application["amt_bur_credit_diff"].fillna(0)
    application["prb_bur"] = application["prb_bur"].fillna(0)
    application["prbs_six_months"] = application["prbs_six_months"].fillna(0)
    application["perc_paym_inst_min"] = application["perc_paym_inst_min"].fillna(0)
    application["perc_inst_min_bal"] = application["perc_inst_min_bal"].fillna(0)
    application["perc_bal_tot_rec"] = application["perc_bal_tot_rec"].fillna(0)
    application["prbs_twelve_months"] = application["prbs_twelve_months"].fillna(0)
    application["perc_paym_inst_min_12"] = application["perc_paym_inst_min_12"].fillna(0)
    application["perc_inst_min_bal_12"] = application["perc_inst_min_bal_12"].fillna(0)
    application["perc_bal_tot_rec_12"] = application["perc_bal_tot_rec_12"].fillna(0)
    application["prbs_three_months"] = application["prbs_three_months"].fillna(0)
    application["perc_paym_inst_min_3"] = application["perc_paym_inst_min_3"].fillna(0)
    application["perc_inst_min_bal_3"] = application["perc_inst_min_bal_3"].fillna(0)
    application["perc_bal_tot_rec_3"] = application["perc_bal_tot_rec_3"].fillna(0)
    application["AMT_CREDIT_POS"] = application["AMT_CREDIT_POS"].fillna(0)
    application["prbs_pos_dpd"] = application["prbs_pos_dpd"].fillna(0)
    application["prbs_pos_dpd_def"] = application["prbs_pos_dpd_def"].fillna(0)
    application["prb_insts_12"] = application["prb_insts_12"].fillna(0)
    application["perc_paym_inst"] = application["perc_paym_inst"].fillna(0)
    application["prb_insts_6"] = application["prb_insts_6"].fillna(0)
    application["prb_insts_3"] = application["prb_insts_3"].fillna(0)
    
    for column in application.columns:
        if application[column].dtype == "object":
            application[column] = application[column].astype('category')

    application = pd.get_dummies(application, dummy_na = True)

    print("remove correlated features")
    tr_corr = application.head(30000)
    
    corr_matrix = tr_corr.corr().abs()
    # Threshold for removing correlated variables
    threshold = 0.99

    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
    if("TARGET" in to_drop):
        to_drop.remove("TARGET")
        
    print('There are %d columns to remove.' % (len(to_drop)))
    application = application.drop(columns = to_drop)
    
    inst_pay = installments_payments.groupby(["SK_ID_CURR", "DAYS_ENTRY_PAYMENT"]).agg({"DAYS_INSTALMENT" : "min", "AMT_INSTALMENT" : "sum", "AMT_PAYMENT" : "sum"}).reset_index()
    inst_pay = inst_pay.sort_values(["SK_ID_CURR", "DAYS_ENTRY_PAYMENT"], ascending = True).reset_index()
    inst_pay["DAYS_INSTALMENT_DIFF"] = inst_pay.groupby("SK_ID_CURR").DAYS_INSTALMENT.diff()
    inst_pay["DAYS_ENTRY_PAYMENT_DIFF"] = inst_pay.groupby("SK_ID_CURR").DAYS_ENTRY_PAYMENT.diff()
    inst_pay["AMT_INSTALMENT_DIFF"] = inst_pay.groupby("SK_ID_CURR").AMT_INSTALMENT.diff()
    inst_pay["AMT_PAYMENT_DIFF"] = inst_pay.groupby("SK_ID_CURR").AMT_PAYMENT.diff()
    inst_pay["AMT_PAYMENT_DIFF_div_DAYS_ENTRY_PAYMENT_DIFF"] = inst_pay["AMT_PAYMENT_DIFF"].divide(inst_pay["DAYS_ENTRY_PAYMENT_DIFF"].replace(0,np.nan))
    inst_pay = inst_pay[["SK_ID_CURR", "AMT_PAYMENT_DIFF", "DAYS_ENTRY_PAYMENT_DIFF"]]
    application = application.join(inst_pay.groupby("SK_ID_CURR").max(), rsuffix = "_max")
    application = application.join(inst_pay.groupby("SK_ID_CURR").min(), rsuffix = "_min")

    train = application.loc[application["TARGET"].notnull()]
    test = application.loc[application["TARGET"].isnull()]
    test = test.drop(columns = ["TARGET"])
    
    print('Training shape: ', train.shape)
    print('Testing shape: ', test.shape)
    
    train.to_csv("processed/train_model_base_1.csv")
    test.to_csv("processed/test_model_base_1.csv")