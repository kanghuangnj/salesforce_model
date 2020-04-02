from model_lib.config import MODEL_SPEC, WEWORK_DIR
import pandas as pd
def test(args):
    batch_size = 1024
    WEWORK_DIR = 'data/wework'
    test = pd.read_csv(os.path.join(WEWORK_DIR, 'ratings_test.dat'), sep=',', header=0, names=['', 'account_id', 'location_id', 'rating'],  engine='python')
    spec = MODEL_SPEC[args.model]
    config = spec['config']
    Engine = spec['engine']
    # Specify the exact model
    engine = Engine(config)
    model = engine.model.eval()
    all_test_accs, all_test_locs, all_test_scores = [], [], []
    labels = []
    for row in test.itertuples():
        all_test_accs.append(int(row.account_id))
        all_test_locs.append(int(row.location_id))
        labels.append(row.rating)
    
    for i in tqdm(range(0, len(all_test_accs) // batch_size+1)):
        test_accs = torch.LongTensor(all_test_accs[i*batch_size: min(len(all_test_accs), (i+1)*batch_size)])
        test_locs = torch.LongTensor(all_test_locs[i*batch_size: min(len(all_test_accs), (i+1)*batch_size)])
        if config['use_cuda'] is True:
            test_accs = test_accs.cuda()
            test_locs = test_locs.cuda()
        test_scores = model(test_accs, test_locs)
        all_test_scores.extend(test_scores.data.view(-1).tolist())
    

    # for i, score in enumerate(all_test_scores):
    #     if score > 0.2:
    #         all_test_scores[i] = 1
    #     else:
    #         all_test_scores[i] = 0
    test_results = {'account_id': all_test_accs, 
                    'location_id': all_test_locs,
                    'prob': all_test_scores,
                    'label': labels}    
    pred_df = pd.DataFrame(test_results)

    pred_df.to_csv('test_pred.csv')