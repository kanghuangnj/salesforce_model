from tqdm import tqdm
from model_lib.config import  WEWORK_DIR, mlp_config
from model_lib.mlp import MLPEngine
from model_lib.data import dataloader
import pandas as pd
import torch 

MODEL_SPEC = {
      'mlp':{
              'config': mlp_config,
              'engine': MLPEngine,
            },
    }

def test(args):
    batch_size = 1024
    spec = MODEL_SPEC[args['model']]
    config = spec['config']
    Engine = spec['engine']
    dataset = dataloader(WEWORK_DIR, config)
    print ('data loaded ...')
    # Specify the exact model
    engine = Engine(config)
    model = engine.model.eval()
    accounts, locations, scores = [], [], []
    
    acc2id = dataset['acc2id']
    loc2id = dataset['loc2id']
    ratings = dataset['rating']
    labels = []
    
    for row in ratings.itertuples():
        if (not row.account_id in acc2id) or (not row.location_id in loc2id): continue
        accounts.append(int(acc2id[row.account_id]))
        locations.append(int(loc2id[row.location_id]))
        labels.append(row.rating)
        
    n = len(ratings)
    print ('%d ratings' % n)
    for i in tqdm(range(0, n // batch_size+1)):
        accounts = torch.LongTensor(accounts[i*batch_size: min(n, (i+1)*batch_size)])
        locations = torch.LongTensor(locations[i*batch_size: min(n, (i+1)*batch_size)])
        if config['use_cuda'] is True:
            accounts = accounts.cuda()
            locations = locations.cuda()
        preds = model(accounts, locations)
        scores.extend(preds.data.view(-1).tolist())
    

    test_results = {'account_id': accounts, 
                    'location_id': locations,
                    'prob': scores,
                    'label': labels}    
    pred_df = pd.DataFrame(test_results)
    print ('coverage: %.2f' % (len(pred_df)*1.0 / (len(ratings))))
    pred_df.to_csv('salesforce_model_prediction_sample.csv')


# args = {
#     'model': 'mlp',
#     'mode': 'test'
# }

# test(args)
