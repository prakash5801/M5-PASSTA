# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:32:29 2020

@author: pp9596
"""
from  datetime import datetime, timedelta
import gc
import os
import numpy as np, pandas as pd
import lightgbm as lgb
import pickle
from sys import platform
from sklearn.preprocessing import OneHotEncoder
import random

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 
        'snap_WI': 'float32', "nday_holiday_Monday":"category"} 

#"standardized_State_CA":"float32","standardized_State_TX":"float32","standardized_State_WI":"float32"}

CAL_DTYPES_1={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32',
        "Lag_pre_post": "int16", "Lag_pre_Week":"int16", "is_weekend":"int16", "week":"int16", 
        "quarter":"int16", "mday":"int16", "tm_w":"int16"}


# read statistics dataset
if platform == "linux" or platform == "linux2":
    PATH = r"/home/ec2-user/m5/data/series_stats.csv"
    DATA_PATH = "/home/ec2-user/m5/data"
elif platform == "win32":
    PATH = r"C:\Users\PP9596\Documents\Bitbucket\M5\data\series_stats.csv"
    DATA_PATH = r"C:\Users\PP9596\Documents\Bitbucket\M5\data"
    

stats_ds = pd.read_csv(PATH)
TYPE_SERIES = stats_ds['Type'].unique() # ['Intermittent', 'Lumpy', 'Erratic', 'Smooth']

## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def demand(x):
  y = x[x!=0]
  return y

def week_of_month(date_value):
    return (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)


def create_calender_feat(saveflag=True):
    ''' Function to create calender related features
    '''
    cal = pd.read_csv(os.path.join(DATA_PATH, "calendar_pks.csv"), dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])    
    
    date_features = {        
        "wday": "weekday",
        "week": "weekofyear",        
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
        #  "ime": "is_month_end",
        #  "ims": "is_month_start",
        }
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in cal.columns:
            cal[date_feat_name] = cal[date_feat_name].astype("int16")
        else:
            cal[date_feat_name] = getattr(cal["date"].dt, date_feat_func).astype("int16")
            
    # "WeekOfMonth"
    if date_feat_name not in cal.columns:
        cal["tm_w"] = (cal['date'].dt.day-1)//7+1    
    
    # Save the calender
    if saveflag:
        cal.to_csv(os.path.join(DATA_PATH, "calendar_pks.csv"), index=False)
    return cal

def create_agg_sales(DATA_PATH, salesFile = "sales_train_validation.csv"):
    dt = pd.read_csv(os.path.join(DATA_PATH, salesFile))
    cal = pd.read_csv(os.path.join(DATA_PATH, "calendar_pks.csv"))
    prices = pd.read_csv(os.path.join(DATA_PATH, "sell_prices.csv"))
    catcols = ['id', 'item_id', 'store_id', 'cat_id']
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")    

    # Create day of year seasonality
    cal["date"]=pd.to_datetime(cal["date"])
    tm_yday_list=[]
    for i in range(cal.shape[0]):
        date_temp = cal["date"].iloc[i]
        tm_yday_list.append(date_temp.timetuple().tm_yday)
    cal["tm_yday"]=tm_yday_list     

    # merge dataset
    dt = dt.merge(cal[["wm_yr_wk", "d", "tm_yday"]], on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
        
    dt_agg = dt[["tm_yday", "sales"]].groupby("tm_yday", as_index=False).agg("mean")
    cal = cal.merge(dt_agg, on= "tm_yday", copy = False)
    cal.rename({"sales":"avg_sales"}, inplace=True)
    cal.to_csv(os.path.join(DATA_PATH, "calendar_pks.csv"), index=False)
    return cal

def create_price_feat(saveflag=True):
    ''' Function to crease static features for price
    '''
    prices = pd.read_csv(os.path.join(DATA_PATH, "sell_prices.csv"))
    
    # Create pricing features
    prices["sell_price_type"]=prices["sell_price"]
    prices["sell_price_type"] = prices["sell_price_type"].astype(str)
    prices["sell_price_type"] = prices["sell_price_type"].apply(lambda x: x[-1])
    prices["sell_price_type_lag"] = prices[["store_id", "item_id", "sell_price_type"]].groupby(["store_id", "item_id"])["sell_price_type"].shift(1)
    prices["pct_price_change"] = prices[["store_id", "item_id", "sell_price"]].groupby(["store_id", "item_id"])["sell_price"].pct_change()
    # Save the calender
    if saveflag:
        prices.to_csv(os.path.join(DATA_PATH, "prices_pks.csv"), index=False)
    return prices


def read_sales_data(DATA_PATH, start_day = 1, tr_last = 1913, melt_dt=False):    
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(os.path.join(DATA_PATH, "sales_train_evaluation.csv"), usecols = catcols + numcols, dtype = dtype)    
    
    if melt_dt:
        dt = pd.melt(dt, id_vars = catcols, 
                     value_vars = [col for col in dt.columns if col.startswith("d_")], 
                     var_name = "d", value_name = "sales")
    return dt


def create_dt(is_train = True, nrows = None, first_day = 1, stype=None, cat_mean_label=False, 
              tr_last=1941, max_lags=60):
    if type(stype) == str:
        stype = [stype]
    
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32", 
                    "sell_price_type":"category", "sell_price_type_lag":"category", "pct_price_change": "float32"}
    
    # Read price dataset
    prices = pd.read_csv(os.path.join(DATA_PATH, "prices_pks.csv"), dtype = PRICE_DTYPES) # sell_prices.csv
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()    
    
    
    cal = pd.read_csv(os.path.join(DATA_PATH, "calendar_pks.csv"), dtype = CAL_DTYPES)
    
    # add event 
    '''
    unique_event_4_5_month = cal.event_name_1[cal.month.isin([4, 5])].unique().astype(str)
    unique_event_4_5_month = list(unique_event_4_5_month[unique_event_4_5_month!="nan"])
    unique_event_4_5_month.append('OrthodoxChristmas')
    '''
    unique_event_4_5_month = ['OrthodoxChristmas']
    for e in unique_event_4_5_month:
        cal[str(e)] = (cal['event_name_1']==e )*1
        
    # Get dates
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    # Start date
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(os.path.join(DATA_PATH, "sales_train_evaluation.csv"), nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    # Filter only series required series
    if stype is not None:                
        ids = stats_ds.id[stats_ds['Type'].isin(list(stype))]
        dt = dt[dt['id'].isin(ids.tolist())]
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
            
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

    # remove observation with sales NA (Reduce observations)
    DROP_COL = ['event_type_1', 'event_name_2', 'event_type_2']
    train_cols = dt.columns[~dt.columns.isin(DROP_COL)]
    dt = dt[train_cols]
    
    ## Replace catagory with overall mean for series    
    if cat_mean_label:
        for col in catcols:
                if col != "id":
                    dt[col] = dt.groupby(col)['sales'].transform('mean')
    
    return dt



def create_ds(START_TRAIN=1, END_TRAIN=1913, P_HORIZON=28, stype=None):
    print("Processing dataset....")
    dt = create_dt(is_train = True, nrows = None, first_day = START_TRAIN, stype=stype, tr_last=END_TRAIN)
    create_static_fea(dt, stat_feat=True)
    create_dynamic_fea(dt)

    # Find dataset
    dt.shape

    # drop na
    dt.dropna(inplace = True)
    dt.shape

    # drop cols
    drop_cols = ["date", "wm_yr_wk", "weekday"]
    train_cols = dt.columns[~dt.columns.isin(drop_cols)]
    dt = dt[train_cols]

    # Create features
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1"] # , "event_name_2", "event_type_1", "event_type_2"]
    useless_cols = ["id", "sales","d"]
    train_cols = dt.columns[~dt.columns.isin(useless_cols)]

    # create dataset
    dt['d'] = dt['d'].str[2:].astype('int16')
    
    # Create train-test split
    train_mask = dt['d']<=(END_TRAIN-P_HORIZON)
    
    # reset the index
    dt.reset_index(drop=True, inplace = True)

    return dt, train_cols, train_mask, cat_feats


def create_static_fea(dt, stat_feat=False):    
    lags = [28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    lag_cols_log1p = [f"log1p_lag_{lag}" for lag in lags ]
    print("INFO: CREATE LAG FEATURE")
    for lag, lag_col, lag_col_log1p in zip(lags, lag_cols, lag_cols_log1p):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)
        dt[lag_col_log1p] = np.log1p(dt[lag_col])
        # dt[f'ewm_{lag}'] = dt.groupby(['id'])[lag_col].transform(lambda x: x.ewm(span=28).mean())

        
    pz = lambda x: sum(x==0)/28.0
    cv2 = lambda x: (np.std(demand(x))/(np.mean(demand(x))**2))
    count_zero = lambda x: (x==0).sum()
    wins = [28]
    print("INFO: CREATE ROLLING FEATURE")
    for win in wins:
        for lag,lag_col in zip(lags, lag_cols):
            print("rmean with lag - ", lag)
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
            # print("rmax with lag - ", lag)
            # dt[f"rmax_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).max())
            print("rvar with lag - ", lag)
            dt[f"rvar_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).var())
            print("rpz with lag - ", lag)
            # dt[f"rpz_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).apply(pz))
            # print("rcv2 with lag - ", lag)
            # dt[f"rcv2_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).apply(cv2))
            # create zero features
            # dt[f"rcountzero_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).apply(count_zero))
    
    # fill zero for statistical features 
    fill_zero_col = ['rpz_28_28', 'rcv2_28_28']
    for f in fill_zero_col:
        print(f)
        if f in dt.columns:
            dt[f] = dt[f].fillna(0)
    
    # merge statiscal features
    if stat_feat:
        print("INFO: MERGING STATICAL FEATURE")
        stat_feat= pd.read_csv(os.path.join(DATA_PATH, "statscompfeat.csv"))
        
        # Remove non-userful columns
        rem_col = ['localsimple_mean1', 'trev_num', 'fluctanal_prop_r1', 'walker_propcross']
        fin_col = list(set(stat_feat.columns) - set(rem_col))
        stat_feat = stat_feat[fin_col]
        
        # Merge dataset
        dt = dt.merge(stat_feat, on= "id", copy = False)  
    return dt

def create_dynamic_fea(dt, lags=[1, 7, 14], wins=[1, 7, 14]):
    lag_cols = [f"lag_{lag}" for lag in lags ]
    print("INFO: CREATE LAG FEATURE")
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    # mean feature
    print("INFO: CREATE ROLLING FEATURE")
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
    return dt


######################### Aggregation function
def add_group_feature(dt, lvl):
    # do data grouping
    # rev_group = ((abs(max(dt['d'])-dt['d']))//lvl)+1
    rev_group = np.ceil((abs(max(dt['d'])-dt['d'])+1)/lvl)
    dt['group'] = ((max(rev_group)-rev_group)+1) + np.ceil(min(dt['d'])/lvl)
    return dt

def create_aggregated_ds(dt, lvl = 7):            
    # Fixed features
    eligible_agg_level = [7, 14, 28]  
    level_features = {} 
    level_features["sales"] = [["sum"], eligible_agg_level]
    level_features["wday"] = [["count"], eligible_agg_level]
    level_features["month"] = [["mean"], eligible_agg_level]
    level_features["year"] = [["mean"], eligible_agg_level]
    level_features["event_name_1"] = [["sum"], eligible_agg_level]
    level_features["snap_CA"] = [["sum"], eligible_agg_level]
    level_features["snap_TX"] = [["sum"], eligible_agg_level]
    level_features["snap_WI"] = [["sum"], eligible_agg_level]
    level_features["Lag_pre_post"] = [["sum"], eligible_agg_level]
    level_features["Lag_pre_Week"] = [["sum"], eligible_agg_level]
    level_features["is_weekend"] = [["sum"], eligible_agg_level]
    level_features["week"] = [["mean"], eligible_agg_level]
    level_features["quarter"] = [["mean"], eligible_agg_level]
    level_features["mday"] = [["mean"], eligible_agg_level]
    level_features["tm_w"] = [["mean"], eligible_agg_level]
    level_features["sell_price"] = [["mean"], eligible_agg_level]
    level_features["lag_28"] = [["sum"], eligible_agg_level]
    level_features["intercept"] = [["sum"], eligible_agg_level]
    level_features["sell_price_type"] = [[pd.Series.mode], eligible_agg_level]
    level_features["sell_price_type_lag"] = [[pd.Series.mode], eligible_agg_level]
    
    # Holiday 
    Holiday_fileds = ['OrthodoxEaster', 'Pesach End', 'Cinco De Mayo', "Mother's day",
                           'MemorialDay', 'NBAFinalsStart', 'Easter', 'OrthodoxChristmas']
    for h in Holiday_fileds:
        level_features[h] = [["sum"], eligible_agg_level]
        
    # level_features["lag_7"] = [["sum"], eligible_agg_level]
    # level_features["lag_14"] = [["sum"], eligible_agg_level]
    # level_features["rmean_28_28"] = [["mean"], eligible_agg_level]
    # level_features["rvar_28_28"] = [["mean"], eligible_agg_level]        
    # level_features["rmean_7_7"] = [["mean"], eligible_agg_level]
    # level_features["rmean_14_7"] = [["mean"], eligible_agg_level]
    # level_features["rmean_7_14"] = [["mean"], eligible_agg_level]
    # level_features["rmean_14_14"] = [["mean"], [7]]
    
    if lvl not in  eligible_agg_level:
        print("ERROR: LEVEL passed not eligible")
        return None
    
    # remove observation to round off to last level
    # n_rem = max(dt['d'])%lvl
    # dt = dt[n_rem:]
    
    # rev_group = ((abs(dt['d'] - max(dt['d'])+1))//lvl)+1
    dt=add_group_feature(dt, lvl) #  dt['d']//lvl +1
    dt['intercept'] = 1
    agg_fun = create_agg_fun(level_features, lvl=lvl)
    
    GROUP_BY_COL = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'group']
    X_df = dt.groupby(GROUP_BY_COL , as_index=False).agg(agg_fun)
    X_df.columns = get_col_name(X_df)
    print("Check before dropping - ", X_df.group.max())
    print("Check before dt - ", dt.d.max())
    
    # remove partial data points
    ix = X_df.intercept==lvl
    print("Removing partial data points - ", sum(~ix))
    X_df = X_df[ix]
    X_df = X_df.drop(['intercept'], 1)    
    
    return X_df


def create_agg_fun(level_features, lvl=11):
    agg_fun={}
    KEYS = list(level_features.keys())
    for i in KEYS:
        operator  = level_features[i][0]
        if lvl in level_features[i][1]:
            agg_fun[i]=operator                                    
    return agg_fun


def get_col_name(X_df_L11):
    ''' Get column names
    '''
    new_col_name = []
    for c in X_df_L11.columns:
        if type(c)==str:
            new_col_name.append(c)
        else:
            new_col_name.append(c[0])
            
    return new_col_name



def create_agg_features(dt, lvl=0):   
    # add frequency parameter yearly
    if lvl==7:
        dt["seasonality"] = dt['d']%52
     
    if lvl==14:
        dt["seasonality"] = dt['d']%26
        
    if lvl==28:
        dt["seasonality"] = dt['d']%13
    return dt