import math
import pandas as pd
from sklearn.metrics import roc_auc_score,  mean_absolute_error, accuracy_score

class MetronAtK(object):
    def __init__(self):
        # self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores, gold_scores = subjects[0], subjects[1], subjects[2], subjects[3]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,   
                             'test_score': test_scores,
                             'gold_score': gold_scores})
        self._subjects = test
    
    def cal_auc(self):
        test = self._subjects
        return roc_auc_score(test['gold_score'], test['test_score'])


    def cal_hit_ratio(self, top_k):
        """Hit Ratio @ top_K"""
        full = self._subjects
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        #return len(test_in_top_k) * 1.0 / full['user'].nunique()
        return test_in_top_k['weight'].sum()

    def cal_ndcg(self, top_k):
        full = self._subjects
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
        #print (test_in_top_k)
        #test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        #return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
        test_in_top_k['ndcg'] = test_in_top_k.apply(lambda x: x['weight'] * math.log(2) / math.log(1 + x['rank']), axis=1) # the rank starts from 1
        return test_in_top_k['ndcg'].sum()