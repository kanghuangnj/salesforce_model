import torch
import os
from model_lib.utils import save_checkpoint, resume_checkpoint, use_optimizer
from model_lib.metrics import MetronAtK
from model_lib.config import WEWORK_DIR
from torch.autograd import Variable
from tensorboardX import SummaryWriter

pj = os.path.join
class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """
    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK()
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        if not config['implicit']:
        # explicit feedback
            self.crit = torch.nn.MSELoss()
        else:
        # implicit feedback
            self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        # print (ratings_pred)
        # print (ratings)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items, gold_scores = evaluate_data[0], evaluate_data[1], evaluate_data[2]
        
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
            test_scores = self.model(test_users, test_items)
            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
            # print (test_scores.data.view(-1).tolist())
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                    test_items.data.view(-1).tolist(),
                                    test_scores.data.view(-1).tolist(),
                                    gold_scores.data.view(-1).tolist()]

        auc = self._metron.cal_auc()
        print('Epoch {}: {}'.format(epoch_id, auc))
        return auc

    def save(self, alias, epoch_id, auc):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = pj(WEWORK_DIR, self.config['model_dir'].format(alias, epoch_id, auc))
        save_checkpoint(self.model, model_dir)

    def load(self, alias, epoch_id, auc):
        model_dir = self.config['model_dir'].format(alias, epoch_id, auc)
        device_id = -1
        if self.config['use_cuda'] is True:
            device_id = config['device_id']
        resume_checkpoint(self.model, model_dir=model_dir , device_id=device_id)
        return self.model