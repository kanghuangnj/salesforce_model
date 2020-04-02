import torch
import random
import pandas as pd
import pickle
import numpy as np
import os
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from model_lib.config import WEWORK_DIR
pj = os.path.join
random.seed(0)
class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, dataset, config):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        self.dataset = dataset
        ratings = dataset['rating']
        self.ratings = ratings
        assert 'account_id' in ratings.columns
        assert 'location_id' in ratings.columns
        assert 'rating' in ratings.columns
        self.mode = None
        if not config['implicit']:
        # explicit feedback using _normalize and implicit using _binarize
            self.preprocess_ratings = self._normalize(ratings)
        else:
            self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['account_id'].unique())

        self.train_ratings, self.testval_ratings = self._split(self.preprocess_ratings)
        self.test_ratings = self.testval_ratings.sample(len(self.testval_ratings) // 2).reset_index(drop=True)
        self.val_ratings = self.testval_ratings[~self.testval_ratings.index.isin(self.test_ratings.index)]

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, implicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split(self, ratings):
        """leave one out train/test split """
        train = ratings.sample(frac=0.8)
        test = ratings[~ratings.index.isin(train.index)]
        return train[['account_id', 'location_id', 'rating']], test[['account_id', 'location_id', 'rating']]

    def instance_a_train_loader(self):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        acc2id = self.dataset['acc2id']
        loc2id = self.data['loc2id']
        batch_size = self.dataset['batch_size']
        for row in self.train_ratings.itertuples():
            users.append(int(acc2id[row.account_id]))
            items.append(int(loc2id[row.location_id]))
            ratings.append(float(row.rating))
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def val(self):
        self.mode = 'val'
    def test(self):
        self.mode = 'test'
    @property
    def evaluate_data(self):
        """create evaluate data"""
        if self.mode == 'test':
            eval_ratings = self.test_ratings
        elif self.mode == 'val':
            eval_ratings = self.val_ratings
        test_users, test_items, gold_scores = [], [], []
        acc2id = self.dataset['acc2id']
        loc2id = self.dataset['loc2id']
   
        for row in eval_ratings.itertuples():
            test_users.append(int(acc2id[row['account_id']]))
            test_items.append(int(loc2id[row['location_id']]))
            gold_scores.append(row['rating'])
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.FloatTensor(gold_scores)]

def dataloader(path, config):
    acc_embedding = np.load(pj(path, config['pre_acc']))
    loc_embedding = np.load(pj(path, config['pre_loc']))
    with open(pj(path, 'loc2id.pkl'), 'rb') as f:
        loc2id = pickle.load(f)
    with open(pj(path, 'acc2id.pkl'), 'rb') as f:
        acc2id = pickle.load(f)
    rating = pd.read_csv(pj(path, 'rating.csv'))
    rating = rating.rename(columns={'atlas_location_uuid': 'location_id', 'label': 'rating'})
    return {
        'account_embedding': acc_embedding,
        'location_embedding': loc_embedding,
        'loc2id': loc2id,
        'acc2id': acc2id,
        'rating': rating,
    }