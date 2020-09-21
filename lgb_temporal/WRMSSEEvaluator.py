# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:06:50 2020

@author: pp9596
"""

from typing import Union
import os
import numpy as np
import pandas as pd

class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                 calendar: pd.DataFrame, prices: pd.DataFrame,
                 val_data_melt=None):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices
        self.val_data_melt = val_data_melt

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(self.group_ids):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y) 
            # setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum()) 
            if type(group_id)==str:
                setattr(self, f'lv{i + 1}_valid_df', valid_df[[group_id]+valid_target_columns].groupby(group_id).sum()) # valid_df.groupby(group_id)[valid_target_columns].sum()
            else:
                setattr(self, f'lv{i + 1}_valid_df', valid_df[group_id+valid_target_columns].groupby(group_id).sum()) # valid_df.groupby(group_id)[valid_target_columns].sum()

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)            
            lv_scores = lv_scores.replace([np.inf, -np.inf], np.nan) # replace inf by nan
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())
        return np.mean(all_scores)
    
class WRMSSEForLightGBM(WRMSSEEvaluator):    
    def feval(self, preds, dtrain):
        preds = pd.DataFrame({"sales":preds})        
        preds = pd.concat([self.val_data_melt.reset_index(drop=True), preds], axis=1)        
        preds_pivot = preds[["id", "d", "sales"]].pivot(index='id', columns='d', values='sales')
        preds_pivot = preds_pivot.fillna(0)
        preds_pivot = preds_pivot.reindex(list(self.train_df.id.values))  
        preds_pivot.columns = self.valid_target_columns
        # print("Sum - ", sum(preds_pivot.iloc[:, -28:].sum()))
        preds_pivot = preds_pivot.reset_index(drop=True)
        score = self.score(preds_pivot.iloc[:, -28:])
        return 'WRMSSE', score, False
    
if __name__=="__main__":
    DATA_PATH = "C:/Users/PP9596/Documents/Bitbucket/M5/data"
    train_df =pd.read_csv(os.path.join(DATA_PATH, "sales_train_validation.csv"))
    
    calendar = pd.read_csv(os.path.join(DATA_PATH, "calendar.csv"))
    calendar["date"] = pd.to_datetime(calendar["date"])
    
    prices = pd.read_csv(os.path.join(DATA_PATH, "sell_prices.csv"))

    train_fold_df = train_df.iloc[:, :-28]
    valid_fold_df = train_df.iloc[:, -28:]
    valid_preds = valid_fold_df.copy() + np.random.randint(100, size=valid_fold_df.shape)
    
    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
    evaluator.score(valid_preds)
    
    # Implementing in LightGBM
    evaluator = WRMSSEForLightGBM(train_fold_df, valid_fold_df, calendar, prices)
    model = lgb.train(params, dtrain,
                  num_boost_round=10000,
                  valid_sets=dvalid,
                  feval=evaluator.feval,
                  early_stopping_rounds=200)
    
    all_scores = []
    for i, group_id in enumerate(evaluator_val.group_ids):
        lv_scores = evaluator_val.rmsse(valid_preds.groupby(group_id)[evaluator_val.valid_target_columns].sum(), i + 1)
        weight = getattr(evaluator_val, f'lv{i + 1}_weight')
        lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
        all_scores.append(lv_scores.sum())