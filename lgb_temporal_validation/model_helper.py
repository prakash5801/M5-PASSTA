# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:53:04 2020

@author: pp9596
"""

from  datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os, sys, warnings, pickle
from multiprocessing import Pool        # Multiprocess Runs
import lightgbm as lgb

# hyperopt set-up
from functools import partial
from hyperopt import hp, tpe, space_eval, STATUS_OK, Trials
from hyperopt.fmin import fmin 



# custom imports
import sys
import s3fs

from data_helper import seed_everything, demand, create_dt, create_static_fea, create_dynamic_fea, create_ds
from data_helper import create_aggregated_ds, create_agg_features, read_sales_data
from WRMSSEEvaluator import WRMSSEEvaluator, WRMSSEForLightGBM
from aws_helper import write_csv_s3, read_csv_s3
warnings.filterwarnings('ignore')


# read statistics dataset
from sys import platform
if platform == "linux" or platform == "linux2":
    DATA_PATH = "/home/ec2-user/m5/data"
    aux_model_path = r"/home/ec2-user/m5/python/aux_model"
    sub_path = r"/home/ec2-user/m5/submission"
    META_DATA_PATH = "/home/ec2-user/m5/data/meta_data"
elif platform == "win32":
    DATA_PATH = r"C:\Users\PP9596\Documents\Bitbucket\M5\data"
    aux_model_path = r"C:\Users\PP9596\Documents\Bitbucket\M5\python\aux_model"
    sub_path = r"C:\Users\PP9596\Documents\Bitbucket\M5\submission"
    

def create_lgb_evaluator(dt, train_mask, valid_mask):
    
    idDF = pd.DataFrame({"id":dt["id"].unique()})
    train_fold_df = dt[train_mask][["id", "d", "sales"]].pivot(index='id', columns='d', values='sales')
    train_fold_df = idDF.merge(train_fold_df, on="id", how='left')
    valid_fold_df = dt[valid_mask][["id", "d", "sales"]].pivot(index='id', columns='d', values='sales')
    valid_fold_df = idDF.merge(valid_fold_df, on="id", how='left')
    # train_fold_df.reset_index(inplace=True)
    # valid_fold_df.reset_index(inplace=True) 
    cols_train = ["id"] + [f"d_{x}" for x in train_fold_df.columns[1:]]
    cols_val = ["id"] + [f"d_{x}" for x in valid_fold_df.columns[1:]]
    train_fold_df.columns = cols_train
    valid_fold_df.columns = cols_val
    valid_fold_df = valid_fold_df.fillna(0)
    train_fold_df = train_fold_df.fillna(0)
    
    ## merge static features in train
    static_df = pd.read_csv(os.path.join(DATA_PATH, "sales_train_validation.csv"))
    static_df = static_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    train_fold_df = train_fold_df.merge(static_df, on="id", copy = False)
    train_fold_df = train_fold_df.reset_index(drop=True)
    valid_fold_df = valid_fold_df.reset_index(drop=True)
    
    calendar = pd.read_csv(os.path.join(DATA_PATH, "calendar.csv"))
    calendar["date"] = pd.to_datetime(calendar["date"])
    prices = pd.read_csv(os.path.join(DATA_PATH, "sell_prices.csv"))
    evaluator = WRMSSEForLightGBM(train_fold_df, valid_fold_df, calendar, prices)
    # valid_preds = valid_fold_df.copy() + np.random.randint(100, size=valid_fold_df.shape)
    # evaluator.score(valid_preds)
    return evaluator

def get_valid_data(validation_flag=False):
    ''' Get dataset for evaluation
    '''
    if validation_flag:
        train_df =pd.read_csv(os.path.join(DATA_PATH, "sales_train_validation.csv"))  
    else:
        train_df =pd.read_csv(os.path.join(DATA_PATH, "sales_train_evaluation.csv"))  
        
    calendar = pd.read_csv(os.path.join(DATA_PATH, "calendar.csv"))
    calendar["date"] = pd.to_datetime(calendar["date"])
    prices = pd.read_csv(os.path.join(DATA_PATH, "sell_prices.csv"))
    return train_df, calendar, prices  


def gen_eval_obj(validation_flag=True, tr_start_data=120, tr_end_date=1885):
    # load data for estimator
    train_df, calendar, prices = get_valid_data(validation_flag)
    train_df.sort_values("id", inplace = True)
    train_df.reset_index(drop=True, inplace = True)

    # create WRMSSE object
    # Load train fold data for WRMSSEE
    catcol = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    tr_numcol = catcol + [f"d_{i}" for i in range(tr_start_data,tr_end_date+1)]
    vl_numcol = [f"d_{i}" for i in range(tr_end_date+1, tr_end_date+29)]    
    # train_fold_df = train_df.iloc[:, :-28]
    # valid_fold_df = train_df.iloc[:, -28:]
    train_fold_df = train_df[tr_numcol]
    valid_fold_df = train_df[vl_numcol]

    # load WRMSSEE class
    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
    valid_col = valid_fold_df.columns
    valid_col = valid_col.insert(0, "id")
    return evaluator, valid_col, train_df["id"]


def gen_eval_obj_val(validation_flag=True, tr_start_data=120, tr_end_date=1885):
    # load data for estimator
    train_df, calendar, prices = get_valid_data(validation_flag)
    train_df.sort_values("id", inplace = True)
    train_df.reset_index(drop=True, inplace = True)

    # create WRMSSE object
    # Load train fold data for WRMSSEE
    catcol = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    tr_numcol = catcol + [f"d_{i}" for i in range(tr_start_data,tr_end_date+1)]
    vl_numcol = [f"d_{i}" for i in range(tr_end_date+1, tr_end_date+29)]    
    # train_fold_df = train_df.iloc[:, :-28]
    # valid_fold_df = train_df.iloc[:, -28:]
    train_fold_df = train_df[tr_numcol]
    valid_fold_df = train_df[vl_numcol]

    # load WRMSSEE class
    # evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
    valid_col = valid_fold_df.columns
    valid_col = valid_col.insert(0, "id")
    return train_fold_df, valid_fold_df, calendar, prices, valid_col


def run_model(dt, modelname, sub_fname, END_TRAIN, P_HORIZON, lgb_params, VER,
              TIME_FILTER=0,  TEMP_AGG_NRUNS=False, logit_model=False, 
              TARGET = "sales", estimator=None, SEED=1010, use_fake_val=False, 
              val_prop=0.2, wrmssee_flag=False):        
    # get training columns
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "nday_holiday_Monday"] # , "event_name_2", "event_type_1", "event_type_2"]
    useless_cols = ["id", "sales","d", "group", "intercept", "avg_sales", "FOOD3_CA_Break_point"]
    train_cols = dt.columns[~dt.columns.isin(useless_cols)] 
    
    if use_fake_val:
        val_sam_size= int(len(dt.index.values)*val_prop)
        fake_valid_inds = np.random.choice(dt.index.values, val_sam_size, replace = False)
        train_inds = np.setdiff1d(dt.index.values, fake_valid_inds)
        train_mask = dt.index.isin(train_inds)
        valid_mask = dt.index.isin(fake_valid_inds)
    else:
        if estimator is not None:
            train_mask = (dt['d']<=END_TRAIN) & (dt['d']>TIME_FILTER)
        else:
            train_mask = (dt['d']<=(END_TRAIN-P_HORIZON)) & (dt['d']>TIME_FILTER)
        valid_mask = dt['d']>(END_TRAIN-P_HORIZON)

    
    # train on optimal param
    ybinary = dt[TARGET].copy()
    ybinary[ybinary>0]=1
    ybinary[ybinary==0]=0
    if logit_model:
        print("Running Logit Model")
        train_data = lgb.Dataset(dt[train_mask][train_cols], label=ybinary[train_mask], categorical_feature=cat_feats)
        valid_data = lgb.Dataset(dt[valid_mask][train_cols], label=ybinary[valid_mask], categorical_feature=cat_feats)
        lgb_params['objective'] ='binary' 

    else:
        train_data = lgb.Dataset(dt[train_mask][train_cols], label=dt[train_mask][TARGET], categorical_feature=cat_feats, free_raw_data=False)
        valid_data = lgb.Dataset(dt[valid_mask][train_cols], label=dt[valid_mask][TARGET], categorical_feature=cat_feats, free_raw_data=False)
        # lgb_params_fin = lgb_params

    if estimator is not None:
        NUM_ITERATION = estimator.best_iteration # Iteration based on validation and test (Re-run the pipeline all 4000 iteration)
        print("Number of iteration: ", estimator.best_iteration)
    else:
        NUM_ITERATION=4000

    lgb_params['seed'] = SEED                       # as possible
    lgb_params['n_estimators'] =NUM_ITERATION       # as possible
    seed_everything(SEED)
    
    if wrmssee_flag:
        tr_start_data = dt['d'].min()
        tr_end_date = END_TRAIN-P_HORIZON
        train_fold_df, valid_fold_df, calendar, prices, valid_col = gen_eval_obj_val(validation_flag=False, 
                                                            tr_start_data=tr_start_data, tr_end_date=tr_end_date)
        evaluator_val = WRMSSEForLightGBM(train_fold_df, valid_fold_df, calendar, prices, dt[valid_mask][["id", "d"]])
        
        estimator = lgb.train(lgb_params,
                  train_data,
                  valid_sets = [valid_data], # fake validation
                  feval=evaluator_val.feval,
                  verbose_eval = 20,
                  early_stopping_rounds = 100)        
    else:
        estimator = lgb.train(lgb_params,
                  train_data,
                  valid_sets = [valid_data], # fake validation
                  verbose_eval = 200,
                  early_stopping_rounds = 50)    
    pickle.dump(estimator, open(os.path.join(aux_model_path , modelname), 'wb'))
    
    # get logit threshold
    sales_hat = estimator.predict(dt[valid_mask][train_cols])
    if logit_model:
        fname = "logit_valid_PRED_" + str(0) + "_" + str(VER)+".csv"
        pred_val = pd.DataFrame({"id":dt[valid_mask]["id"], "label":ybinary[valid_mask], "actual_sales":dt[valid_mask][TARGET], "pred":sales_hat})        
    else:
        fname = "reg_valid_PRED_" + str(0) + "_" + str(VER)+".csv"
        pred_val = pd.DataFrame({"id":dt[valid_mask]["id"], "label":ybinary[valid_mask], "actual_sales":dt[valid_mask][TARGET], "pred":sales_hat})        
    pred_val.to_csv(os.path.join(sub_path, fname),index=False)        
    return train_cols, pred_val


def get_lgbm_varimp(estimator, max_vars=None, ascendFlag=True):
    cv_varimp_df = pd.DataFrame([estimator.feature_name(), estimator.feature_importance()]).T
    cv_varimp_df.columns = ['feature_name', 'varimp']
    cv_varimp_df.sort_values(by='varimp', ascending=ascendFlag, inplace=True)  
    if max_vars is not None:
        cv_varimp_df = cv_varimp_df.iloc[0:max_vars]
    return cv_varimp_df   

def save_result(col, te_sub, VER, PRED_DURATION=28, save_sub=False):
    last_col = "".join(["F", str(PRED_DURATION)])
    print("Total Sum - ", sum(te_sub.loc[:, 'F1':last_col].sum()))
    sub = te_sub
    sub2 = sub.copy()
    if sub2["id"].str.contains("validation").sum()>0:
        sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    else:
        sub2["id"] = sub2["id"].str.replace("evaluation$", "validation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    fname = 'lgb_model_' + col + '_v'+str(VER)+ "_modelrun_" + '.csv' 
    if save_sub:
        sub.to_csv(os.path.join(sub_path, fname),index=False)
    return sub

def grouping_level(te_sub, q=0.05, PRED_DURATION=28):
    # Group by aggregation
    level_group_by={}
    level_group_by["L11"] = ["state_id", "item_id"]
    level_group_by["L10"] = ["item_id"]
    level_group_by["L9"] = ["store_id", "dept_id"]
    level_group_by["L8"] = ["store_id", "cat_id"]
    level_group_by["L7"] = ["state_id", "dept_id"]
    level_group_by["L6"] = ["state_id", "cat_id"]
    level_group_by["L5"] = ["dept_id"]
    level_group_by["L4"] = ["cat_id"]
    level_group_by["L3"] = ["store_id"]
    level_group_by["L2"] = ["state_id"]
    level_group_by["L1"] = ["d"]
    
        
    LEVELS = list(level_group_by.keys())
    cols = [f"F{i}" for i in range(1,PRED_DURATION+1)]
    FIN_COL = ["id"]+cols
    solutionDF = pd.DataFrame()
    for lvl in reversed(LEVELS):
        level_widthout_d = level_group_by[lvl]
        if lvl=='L1':
            dftemp = te_sub[cols].sum(axis=0).to_frame()
            dftemp.reset_index(inplace=True)
            dftemp.columns = ["coln", "sales"]
            dftemp["id"] = "".join(["Total_X_", str(f"{q:.3f}"), "_validation"])
            dftemp = dftemp.pivot(index='id', columns='coln', values='sales')  
            dftemp.reset_index(inplace=True)
            solutionDF = dftemp[FIN_COL]                   
        else:
            dftemp = te_sub.groupby(level_group_by[lvl], as_index=False)[cols].sum()
            if len(level_widthout_d)==1:
                dftemp["id"] = [lev1 + "_X_" + str(f"{q:.3f}") + "_validation" for lev1 in dftemp[level_widthout_d[0]].values]                     
            else:            
                dftemp["id"] = [lev1 + "_" + lev2 + "_" + str(f"{q:.3f}") + "_validation" for lev1,lev2 in zip(dftemp[level_widthout_d[0]].values,dftemp[level_widthout_d[1]].values)]         
            solutionDF = solutionDF.append(dftemp[FIN_COL])
    te_sub["id"] = te_sub["id"].str.replace("validation$", "".join([str(f"{q:.3f}"), "_validation"]))
    solutionDF = solutionDF.append(te_sub[FIN_COL])    
    return solutionDF

def save_fin_sub_CI(modelname, sub_fname, train_cols, VER, lvl=0, PRED_DURATION=28, fday=1913, save_sub=False, stype=None, alpha=0.05):
    # load model
    estimator = pickle.load(open(os.path.join(aux_model_path , modelname), 'rb'))
        
    # Scoring state & test static variable
    # fday = datetime(2016,4, 25)
    if lvl==0:
        max_lags = 60
    else:
        max_lags = 5
    
    # compute prediction
    te = create_dt(is_train = False, nrows = None, first_day = 1, stype=stype)
    create_static_fea(te, stat_feat=True)
    te['d'] = te['d'].str[2:].astype('int16') # create dataset
    
    if lvl!=0:
        te = create_aggregated_ds(te, lvl = lvl)
        te = te.rename(columns={"group": "d"}) # update column names                    
        te = create_agg_features(te, lvl=lvl)

    
    for tdelta in range(0, PRED_DURATION):
        print('Predict | Day:', tdelta+1)
        day = fday + tdelta+1 # timedelta(days=tdelta)
        tst = te[(te.d >= day - max_lags) & (te.d <= day)].copy()
        if lvl==0:            
            tst = create_dynamic_fea(tst, lags=[1, 7, 14], wins=[1, 7, 14])
            train_cols = train_cols[train_cols.isin(tst.columns)]
        elif lvl==7:
            tst = create_dynamic_fea(tst, lags=[1, 2], wins=[1, 2])
        elif lvl==14:
            tst = create_dynamic_fea(tst, lags=[2], wins=[2])
        elif lvl==28:
            pass
        else:
            print("INFO: level not found")
            
        tst = tst.loc[tst.d == day, train_cols]
        # allocate prediction 
        te.loc[te.d == day, "sales"] = estimator.predict(tst[train_cols]) 
    
    # create submission
    cols = [f"F{i}" for i in range(1,PRED_DURATION+1)]
    te_sub = te.loc[te.d > fday, ["id", "sales"]].copy()
    
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    
    # create features for aggregation    
    sales_df = read_sales_data(DATA_PATH, start_day = 1, tr_last = 1913, melt_dt=False) 
    
    # te_sub["id"] = te_sub["id"].str.replace("validation$", "")
    sales_df["id"] = sales_df["id"].str.replace("evaluation$", "validation")
    
    te_sub = te_sub.merge(sales_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']], on = "id", copy = False)
    te_sub_ci = grouping_level(te_sub, q=alpha, PRED_DURATION=PRED_DURATION)
    return te_sub_ci
        

def save_fin_sub(modelname, sub_fname, train_cols, VER, lvl=0, PRED_DURATION=28, fday=1913, save_sub=False, stype=None):
    # load model
    estimator = pickle.load(open(os.path.join(aux_model_path , modelname), 'rb'))
        
    # Scoring state & test static variable
    # fday = datetime(2016,4, 25)
    if lvl==0:
        max_lags = 60
    else:
        max_lags = 5
    
    # compute prediction
    te = create_dt(is_train = False, nrows = None, first_day = 1, tr_last=fday, stype=stype)
    create_static_fea(te, stat_feat=True) 
    te['d'] = te['d'].str[2:].astype('int16') # create dataset
    
    if lvl!=0:
        te = create_aggregated_ds(te, lvl = lvl)
        te = te.rename(columns={"group": "d"}) # update column names                    
        te = create_agg_features(te, lvl=lvl)

    
    for tdelta in range(0, PRED_DURATION):
        print('Predict | Day:', tdelta+1)
        day = fday + tdelta+1 # timedelta(days=tdelta)
        tst = te[(te.d >= day - max_lags) & (te.d <= day)].copy()
        if lvl==0:            
            tst = create_dynamic_fea(tst, lags=[1, 7, 14], wins=[1, 7, 14])
            train_cols = train_cols[train_cols.isin(tst.columns)]
        elif lvl==7:
            tst = create_dynamic_fea(tst, lags=[1, 2], wins=[1, 2])
        elif lvl==14:
            tst = create_dynamic_fea(tst, lags=[2], wins=[2])
        elif lvl==28:
            pass
        else:
            print("INFO: level not found")
            
        tst = tst.loc[tst.d == day, train_cols]

        # allocate prediction 
        te.loc[te.d == day, "sales"] = estimator.predict(tst[train_cols]) 

    # create submission
    cols = [f"F{i}" for i in range(1,PRED_DURATION+1)]
    te_sub = te.loc[te.d > fday, ["id", "sales"]].copy()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0., inplace = True)
    te_sub.sort_values("id", inplace = True)
    te_sub.reset_index(drop=True, inplace = True)
    
    # save submission
    te_sub = save_result(col=sub_fname, te_sub = te_sub, VER=VER, PRED_DURATION=PRED_DURATION, save_sub=save_sub)  
    
    # print important variables
    cv_varimp_df = get_lgbm_varimp(estimator, max_vars=None, ascendFlag=True)
    print(cv_varimp_df)

    return te_sub

def get_sum(te_sub, PRED_DURATION=28, inc_eval=False):

    last_col = "".join(["F", str(PRED_DURATION)])
    if inc_eval:
        ix = te_sub.id.str.contains("validation") | te_sub.id.str.contains("evaluation")
        sum_val = sum(te_sub.loc[ix, 'F1':last_col].sum())
    else:
        ix = te_sub.id.str.contains("validation")
        sum_val = sum(te_sub.loc[ix, 'F1':last_col].sum())    
    print("Total Sum - ", sum_val)
    return sum_val