
from feature_lib.utils import dataloader
from feature_lib.header import feature_columns
import pandas as pd
import numpy as np
class Feature:
    def __init__(self, source):
        self.df = dataloader(source)
        self.mean = []
        self.std = []
        self.max = []
        self.min = []
       

    def __fit_continuous(self):
        cont_col = self.feature_column['continuous']
        df = self.df
        for col in cont_col:
            col_feat = df[df[col].notna()][col]
            self.mean.append(col_feat.mean())
            self.std.append(col_feat.std())
        self.mean = np.expand_dims(self.mean, axis=0)
        self.std = np.expand_dims(self.std, axis=0)

    def __fit_discrete(self):
        disc_col = self.feature_column['discrete']
        df = self.df
        df[disc_col] = df[disc_col].astype(str)
        discrete_dats = []
        self.history_col_val = {}
        for col in disc_col:
            discrete_dat = pd.get_dummies(df[col], prefix=col, dummy_na=False)  # it has nan itself
            discrete_dats.append(discrete_dat)
            self.history_col_val[col] = df[~df.duplicated(col)][col]
        discrete_dats = pd.concat(discrete_dats, axis=1)
        self.discrete_col_order = discrete_dats.columns.to_list()

    def __transform_continuous(self, df):
        cont_col = self.feature_column['continuous']
        df = df[cont_col]
        df = df.fillna(value=0).astype(float)
        np_dat = df.loc[:,cont_col].to_numpy()
        np_dat = (np_dat - self.mean) / self.std
        return np_dat

    def __transform_discrete(self, df):
        disc_col = self.feature_column['discrete']
        df[disc_col] = df[disc_col].astype(str)
        disc_dats = []
        for col in disc_col:
            history_row = self.history_col_val[col]
            disc_dat = pd.get_dummies(pd.concat([history_row, df[col]], axis=0), prefix=col, dummy_na=False)  # it has nan itself
            disc_dats.append(disc_dat.iloc[len(history_row):])
    
        disc_dats = pd.concat(disc_dats, axis=1)
        np_dat = disc_dats[self.discrete_col_order].to_numpy()
        return np_dat

    def fit(self):
        if 'continuous' in self.feature_column:
            self.__fit_continuous()
        if 'discrete' in self.feature_column:
            self.__fit_discrete()
    
    def transform(self, df):
        np_dats = []
        if 'continuous' in self.feature_column:
            np_dats.append(self.__transform_continuous(df))
        if 'discrete' in self.feature_column:
            np_dats.append(self.__transform_discrete(df))
       
        return np.hstack(np_dats) 

class Account(Feature):
    def __init__(self):
        self.feature_column = feature_columns['account']
        Feature.__init__(self, 'account')


class Location_Scorecard(Feature):
    def __init__(self):
        self.feature_column = feature_columns['location_scorecard']
        Feature.__init__(self, 'location_scorecard')
        ls_df = self.df
        building_df = dataloader('building', rename_flag=True) 
        us_building_df = building_df[(building_df['country'] == 'USA') &
                        (~building_df['atlas_location_uuid'].isna()) & 
                        (building_df['atlas_location_uuid'] != 'TestBuilding')]
        self.df = ls_df[ls_df['atlas_location_uuid'].isin(us_building_df['atlas_location_uuid'])]

class Building(Feature):
    def __init__(self):
        self.feature_column = feature_columns['building']
        Feature.__init__(self, 'building')
        building_df = self.df
        us_building_df = building_df[(building_df['Country__c'] == 'USA') &
                                (~building_df['UUID__c'].isna()) & 
                                (building_df['UUID__c'] != 'TestBuilding')]
        self.df = us_building_df
# feat = Feature('account')
# feat.fit()
# np_dat = feat.transform()
# print (np_dat[0])