mlp_config = {'alias': 'mlp-implicit',
              'num_epoch': 2,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'layers': [761,256,96,32,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 1e-8,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'device_id': -1,
              'pretrain': False,
              'pretrain_mlp': 'checkpoints/{}'.format('mlp-implicit.model'),
              'model_dir': 'checkpoints/{}_epoch{}_auc{:.4f}.model',
              'pre_acc': 'acc_feat.npy',
              'pre_loc': 'loc_feat.npy',
              'mode': 'train',
              'implicit': False}


WEWORK_DIR = '/Users/kanghuang/Documents/work/location_recommendation/salesforce_model/artifacts'