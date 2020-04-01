import pandas as pd
import numpy as np
import math
import os 
import itertools
from feature_lib.header import feature_columns, feature_mappings, datapaths

def union_column(df, tgt_col, src_col):
    # mask1 = (~df[tgt_col].isna()) & (~df[src_col].isna())
    # mask2 = df[tgt_col] != df[src_col]
    # mask = mask1 & mask2
    mask = df[tgt_col].isna() & (~df[src_col].isna())
    df.loc[mask, tgt_col] = df[mask][src_col]
    return df

def filter_na(df, cols):
    mask = df[cols[0]].isna()
    for i in range(1, len(cols)):
        mask &= df[cols[i]].isna()
    return df[~mask]


def dataloader(source, rename_flag=False):
    filepath = datapaths[source]
    df = pd.read_csv(filepath)
    if source in feature_columns:
        feature = list(itertools.chain(*feature_columns[source].values()))
        df = df[feature]
    if rename_flag and source in feature_mappings:
        df = df.rename(columns=feature_mappings[source]).reset_index(drop=True)
    df = df.reset_index()
    return df