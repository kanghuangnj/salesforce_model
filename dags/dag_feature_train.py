from __future__ import print_function

import datetime
import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator,BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator

import os,sys
pj = os.path.join
LIBPATH = pj(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, LIBPATH)
from feature_lib.function import feat_function, merge_feat, rating_gen
from model_lib.train import train

args = {
    'owner': 'kang',
    'depends_on_past': False,
    'start_date': datetime.datetime(2020, 3, 12),
}

dag_id = 'salesforce_feature_generation'
tables = ['account', 'location_scorecard', 'building']

"""
Create a DAG to execute tasks
"""
dag = DAG(
    dag_id=dag_id,
    default_args=args,
    schedule_interval=None,
)


main_op = DummyOperator(
    task_id = 'main_entrance',
    dag= dag,
)


rating_op = PythonOperator(
    task_id='rating',
    python_callable=rating_gen,
    dag=dag,
)

merging_op = DummyOperator(
    task_id = 'merge_all_features',
    dag= dag,
)
# merging_op = PythonOperator(
#     task_id='merge_all_features',
#     provide_context=True,
#     python_callable=merge_feat,
#     op_kwargs={'names': tables},
#     trigger_rule = 'all_done',
#     dag=dag,
# )

train_op = PythonOperator(
    task_id='train',
    python_callable=train,
    trigger_rule = 'all_success',
    dag=dag,
)
# save_op = PythonOperator(
#     task_id='save',
#     provide_context=True,
#     python_callable=save_feat,
#     trigger_rule = 'success',
#     dag=dag,
# )
main_op >> rating_op >> merging_op
feat_ops = {}
for name in feat_function:
    feat_ops[name] = PythonOperator(
                            task_id=name,
                            python_callable=feat_function[name],
                            dag=dag,
                        )
    main_op >> feat_ops[name] >> merging_op
merging_op >> train_op