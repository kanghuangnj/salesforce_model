import torch
import os
import numpy as np
from torch import nn
from model_lib.engine import Engine
from model_lib.utils import use_cuda, resume_checkpoint
from model_lib.config import WEWORK_DIR

class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        num_users, num_items, acc_dim, loc_dim = self.init_embedding()
        self.net_interaction = nn.Sequential(
            nn.Linear(acc_dim+loc_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.Linear(64, 16),
            # nn.BatchNorm1d(16),
            # nn.Dropout(p=0.2)
            # nn.LeakyReLU(),
        )
        
        self.net_id = nn.Sequential(
            nn.Linear(num_items, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            # nn.Linear(32, 8),
        )
        
        self.net_shared = nn.Sequential(
            nn.Linear(64+32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, out_features=1)
        )    

    def forward(self, user_indices, item_indices):
        account_embedding = self.account_embedding(user_indices)
        location_embedding = self.location_embedding(item_indices)
        location_id_embedding = self.location_id_embedding(item_indices) 
        account_location_embedding = torch.cat([account_embedding, location_embedding], dim=-1)  # the concat latent vector
        
        location_id_embedding = self.net_id(location_id_embedding)
        account_location_embedding = self.net_interaction(account_location_embedding)
        vector = torch.cat([account_location_embedding, location_id_embedding], dim=-1)
        logits = self.net_shared(vector)
        #rating = self.logistic(logits)
        return logits

    def init_embedding(self):
        acc_vectors = np.load(self.config['pretrain_acc'])
        loc_vectors = np.load(self.config['pretrain_loc'])
        print (acc_vectors.shape, loc_vectors.shape)
        num_users, acc_dim = acc_vectors.shape
        num_items, loc_dim = loc_vectors.shape
        self.config.update({
            'num_users': num_users,
            'num_items': num_items,
            'acc_dim': acc_dim,
            'loc_dim': loc_dim,
        })
        self.account_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=acc_dim)
        self.location_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=loc_dim)
        self.location_id_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_items)
        if self.config['pretrain']:
            self.account_embedding.weight = nn.Parameter(torch.FloatTensor(acc_vectors))
            self.location_embedding.weight = nn.Parameter(torch.FloatTensor(loc_vectors))
            self.account_embedding.require = False
            self.location_embedding.require = False
            self.location_id_embedding.require = True
        return num_users, num_items, acc_dim, loc_dim


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        if config['pretrain']:
            #self.model.load_pretrain_weights()
            resume_checkpoint(self.model, model_dir=config['pretrain_mlp'], device_id=config['device_id'])
        super(MLPEngine, self).__init__(config)
        print(self.model)

       
