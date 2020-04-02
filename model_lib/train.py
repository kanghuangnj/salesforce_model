
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from model_lib.data import SampleGenerator, dataloader
from model_lib.config import WEWORK_DIR, mlp_config
from model_lib.mlp import MLPEngine


MODEL_SPEC = {
  'mlp':{
    'config': mlp_config,
    'engine': MLPEngine,
  },
}

def flatten(df):
    negatives = []
    for index, row in df.iterrows():
      for negative in row['negatives']:
          negatives.append([row['account_id'], negative])
    return pd.DataFrame(negatives, columns=['account_id', 'negative_id'], dtype=np.int32)



def train(args):
    # Load Data
    # wework_rating = pd.read_csv(os.path.join(WEWORK_DIR, 'ratings.dat'), sep=',', header=0, names=['','account_id', 'location_id', 'rating', 'timestamp', 'weight'],  engine='python')
    # # Reindex
    # account_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    # account_id['userId'] = np.arange(len(user_id))
    # ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    # item_id = ml1m_rating[['mid']].drop_duplicates()
    # item_id['itemId'] = np.arange(len(item_id))
    # ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    # ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

    spec = MODEL_SPEC[args['model']]
    config = spec['config']
    Engine = spec['engine']
    dataset = dataloader(WEWORK_DIR, config)
    dataset.update({
        'batch_size': config['batch_size']
    })
    # DataLoader for training
    sample_generator = SampleGenerator(dataset, config)
    sample_generator.test()
    test_data = sample_generator.evaluate_data
    sample_generator.val()
    val_data = sample_generator.evaluate_data
    # Specify the exact model
    engine = Engine(config)
    best_metric = float('inf')
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(dataset)
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        auc = engine.evaluate(val_data, epoch_id=epoch)
        if auc < best_metric:
            best_epoch = epoch
            best_metric = auc
            engine.save(config['alias'], epoch, auc)
            print ('Epoch {}: found best results on validation data: auc = {:.4f}'.format(epoch, auc))

    engine.load(config['alias'], best_epoch, auc)
    auc = engine.evaluate(test_data, epoch_id=epoch)
    print('Best Epoch {}:  auc = {:.4f}'.format(best_epoch, auc))


args = {
    'model': 'mlp',
    'mode': 'train'
}

train(args)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     arg = parser.add_argument
#     arg('--model', choices=['gmf', 'mlp', 'neumf'])
#     arg('--mode', choices=['train', 'test'])
    
#     args = parser.parse_args()
#     if args.mode == 'train':
#         train(args)
#     elif args.mode == 'test':
#         test(args)