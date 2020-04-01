
import os

from feature_lib.utils import union_column, filter_na, dataloader
from feature_lib.rating import Sampler
from feature_lib.header import OUTPUTPATH, datapaths
from feature_lib.feature import Account, Location_Scorecard, Building
import pandas as pd

import numpy as np
import pickle

pj = os.path.join

def filter_tables():
    source_list = ['location_scorecard', 'account', 'building']#['location_scorecard', 'account', 'geography', 'building', 'opportunity', 'company', 'tour']
    data = {}
    for source, filepath in datapaths.items():
        if not source in source_list: continue
        df = dataloader(source, rename_flag=True)
        data[source] = df

    # acc_df = df_pool['account'].merge(df_pool['company'], how='left',on ='company_id', suffixes=('acc', 'com'))
    # # tgt_col, src_col = 'Unomy_Estimated_Revenue_Formula__c', 'Revenue__c'
    # acc_df = union_column(acc_df, 'unomy_revenue', 'revenue')
    # acc_df = union_column(acc_df, 'unomy_company_size', 'company_size')
    # acc_df = union_column(acc_df, 'unomy_industry', 'industry')
   
    building_df = data['building']
    us_building_df = building_df[(building_df['Country__c'] == 'USA') &
                            (~building_df['UUID__c'].isna()) & 
                            (building_df['UUID__c'] != 'TestBuilding')]
    ls_df = data['location_scorecard']
    ls_df = ls_df[ls_df['atlas_location_uuid'].isin(us_building_df['atlas_location_uuid'])]
    #filter trash account
    target_cols = ['headcount', 'unomy_revenue']
    acc_df = data['account']
    acc_df = filter_na(acc_df, target_cols)
    
    valid_loc = set()
    for loc in ls_df['atlas_location_uuid'].unique():
        valid_loc.add(loc)
    valid_acc = set()
    for acc in acc_df['account_id'].unique():
        valid_acc.add(acc)

    metadata = {
        'valid_loc': valid_loc,
        'valid_acc': valid_acc,
    }
    return metadata

    # rating_location = data.atlas_location_uuid.unique()
    # rating_account = data.account_id.unique()
    # ww_location = ls_df.atlas_location_uuid.unique()
    # ww_account = account_df.account_id.unique()
    # location = set.intersection(set(rating_location),set(ww_location))
    # account = set.intersection(set(rating_account), set(ww_account))
    # data = data[data.account_id.isin(account) & data.atlas_location_uuid.isin(location)]
    # account = data.account_id.unique()
    # location = data.atlas_location_uuid.unique()
    # target_ls_df = ls_df[ls_df.atlas_location_uuid.isin(location)]
    # target_account_df = account_df[account_df.account_id.isin(account)]
    # print (len(data) / len(target_account_df)

def rating_gen():
    data = {}
    source_list = ['opportunity', 'geography', 'building']
    for source, filepath in datapaths.items():
        if source in source_list: continue
        data[source] = dataloader(source, rename_flag=True)

    # op_df = data['opportunity']
    # ls_df = data['location_scorecard'] 
    # acc_df = data['account']
    # building_df = data['building']
    
    sampler = Sampler(data)
    neg_op_df = sampler.negative_sampling()
    pos_op_df = sampler.positive_sampling()
    pos_op_df = pos_op_df[['account_id', 'atlas_location_uuid']]
    neg_op_df = neg_op_df[['account_id', 'atlas_location_uuid']]
    pos_op_df['label'] = 1
    neg_op_df['label'] = 0
    rating_df = pd.concat([pos_op_df, neg_op_df], axis=0)
    rating_df = rating_df.sample(frac=1).reset_index(drop=True)
    
    rating_df.to_csv(pj(OUTPUTPATH, 'rating.csv'))


def define_feat_fit(name):
    def execute_feat_fit():
        class_name = '_'.join(list(map(lambda parts: parts[0].upper() + parts[1:], name.split('_')))) 
        Feature = eval(class_name)
        feat = Feature()
        feat.fit()
        return feat
    return execute_feat_fit


def location_feat_merge(feat_fit):
    ls_df =  feat_fit['location_scorecard'].df
    building_df = feat_fit['building'].df
    loc_set = set()
    ls_list = ls_df['atlas_location_uuid'].unique()
    building_list = building_df['UUID__c'].unique()
    for loc in ls_list:
        if loc in building_list:
            loc_set.add(loc)
    loc_list = list(loc_set)
    ls_df = ls_df.set_index('atlas_location_uuid')
    building_df = building_df.set_index('UUID__c')
    ls_df = ls_df.loc[loc_list]
    building_df = building_df.loc[loc_list]
    loc2id = {}
    for i, loc in enumerate(loc_list):
        loc2id[loc] = id
    building_np = feat_fit['building'].transform(building_df)
    ls_np = feat_fit['location_scorecard'].transform(ls_df)
    loc_feat = np.hstack([building_np, ls_np])
    return loc_feat, loc2id

def account_feat_merge(feat_fit):
    rating_df = pd.read_csv(pj(OUTPUTPATH, 'rating.csv'))
    valid_acc = rating_df.account_id.unique()
    acc_df = feat_fit.df
    acc_df = acc_df[acc_df['Id'].isin(valid_acc)]
    acc_np = feat_fit['account'].transform(acc_df)
    acc2id = {}
    acc_list = acc_df['Id'].unique()
    for i, acc in enumerate(acc_list):
        acc2id[acc] = id
    acc_feat = acc_np
    return acc_feat, acc2id

def merge_feat(names, **context):
    feat_fit = {}
    try:
        for name in names:
            feat_fit[name] = context['task_instance'].xcom_pull(task_ids=name)
    except:
        feat_fit = context['feat_fit']
    loc_feat, loc2id = location_feat_merge(feat_fit)
    acc_feat, acc2id = account_feat_merge(feat_fit)
    np.savetxt('loc_feat.out', loc_feat)
    np.savetxt('acc_feat.out'. acc_feat)
    np.save(pj(OUTPUTPATH, 'loc_feat.out'), loc_feat)
    np.save(pj(OUTPUTPATH, 'acc_feat.out'), acc_feat)
    pickle.dump(loc2id, pj(OUTPUTPATH, 'loc2id.pkl'))
    pickle.dump(acc2id, pj(OUTPUTPATH, 'acc2id.pkl'))
    # features = {
    #     'loc_feat': loc_feat,
    #     'loc2id': loc2id,
    #     'acc_feat': acc_feat,
    #     'acc2id': acc2id
    # }
    # return features

# def save_feat(**context):
#     features = context['task_instance'].xcom_pull(task_ids='merge_all_features')
#     rating_df  = context['task_instance'].xcom_pull(task_ids='rating')

   

names = ['account', 'location_scorecard', 'building']
feat_function = {}
for name in names:
    feat_function[name] = define_feat_fit(name)

feat_fit = {
    'account': feat_function['account'](),
    'building': feat_function['building'](),
    'location_scorecard': feat_function['location_scorecard'](),
    'rating': rating_gen()
}
context = {
    'feat_fit': feat_fit,
    'data': data
}
merge_feat(names, **context)