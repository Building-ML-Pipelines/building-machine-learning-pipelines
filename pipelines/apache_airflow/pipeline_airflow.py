# https://github.com/tensorflow/tfx/blob/master/tfx/examples/airflow_workshop/setup/dags/taxi_pipeline_solution.py

import os
import sys
import datetime

from typing import List, Text

from tfx.orchestration import metadata, pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from base_pipeline import init_components


pipeline_name = 'consumer_complaint_pipeline_airflow'
airflow_dir = os.path.join(os.environ['HOME'], 'airflow')
data_dir = os.path.join(airflow_dir, 'data/consumer_complaints')
module_file = os.path.join(airflow_dir, 'dags/module.py')

pipeline_root = os.path.join(airflow_dir, 'tfx', pipeline_name)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')
serving_model_dir = os.path.join(pipeline_root, 'serving_model', pipeline_name)

airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2020, 4, 17),
}


def init_pipeline(components, pipeline_root:Text, direct_num_workers:int) -> pipeline.Pipeline:
    
    beam_arg = [
        f'--direct_num_workers={direct_num_workers}',
    ]
    p = pipeline.Pipeline(pipeline_name=pipeline_name,
                          pipeline_root=pipeline_root,
                          components=components,
                          enable_cache=True,
                          metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
                          beam_pipeline_args=beam_arg)
    return p




components = init_components(data_dir, module_file, serving_model_dir, 
                                training_steps=100, eval_steps=100)
pipeline = init_pipeline(components, pipeline_root, 0)
DAG = AirflowDagRunner(AirflowPipelineConfig(airflow_config)).run(pipeline)
