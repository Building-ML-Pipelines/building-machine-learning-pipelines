import os
from typing import Dict, Text

import absl
import tensorflow_model_analysis as tfma

from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import CsvExampleGen
from tfx.components.base import executor_spec
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input

# FLAGS = flags.FLAGS
# flags.DEFINE_bool('distributed_training', False,
#                   'If True, enable distributed training.')

pipeline_name = 'consumer_complaint_pipeline_gcp_cloud_ai'

# temp yaml file for Kubeflow Pipelines
output_filename = f"{pipeline_name}.yaml"
output_dir = os.path.join(os.getcwd(), 'pipelines', 'gcp_cloud_ai', 'argo_pipeline_files')

# Directory and data locations (uses Google Cloud Storage).
input_bucket = 'gs://consumer_complaint_gcp_cloud_ai'
output_bucket = 'gs://consumer_complaint_gcp_cloud_ai'
data_dir = module_file = os.path.join(input_bucket, 'data')

tfx_root = os.path.join(output_bucket, 'tfx')
pipeline_root = os.path.join(tfx_root, pipeline_name)

# Google Cloud Platform project id to use when deploying this pipeline.
project_id = 'oreilly-book'  # <--- needs update by the user

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
# Copy this from the current directory to a GCS bucket and update the location
# below.
module_file = os.path.join(input_bucket, 'components', 'module.py')

# Region to use for Dataflow jobs and AI Platform jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
gcp_region = 'us-central1'

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
ai_platform_training_args = {
    'project': project_id,
    'region': gcp_region,
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use a
    # a public container image matching the installed version of TFX.
    'masterConfig': { 'imageUri': 'gcr.io/oreilly-book/ml-pipelines-tfx-custom:0.21.3'},  # <---- @Catherine: This might not work for you! You might have to recreate the image on your side
    # Note that if you do specify a custom container, ensure the entrypoint
    # calls into TFX's run_executor script (tfx/scripts/run_executor.py)  <--- Important
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
ai_platform_serving_args = {
    'model_name': 'consumer_complaint',
    'project_id': project_id,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    'regions': [gcp_region],
}


# def _create_pipeline(
#     pipeline_name: Text, pipeline_root: Text, module_file: Text,
#     ai_platform_training_args: Dict[Text, Text],
#     ai_platform_serving_args: Dict[Text, Text]) -> pipeline.Pipeline:
#   """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines."""


beam_pipeline_args = [
    '--runner=DataflowRunner',
    '--experiments=shuffle_mode=auto',
    '--project=' + project_id,
    '--temp_location=' + os.path.join(output_bucket, 'tmp'),
    '--region=' + gcp_region,
    '--disk_size_gb=50',
]

# Number of epochs in training.
train_steps = data_types.RuntimeParameter(
    name='train-steps',
    default=10000,
    ptype=int,
)

# Number of epochs in evaluation.
eval_steps = data_types.RuntimeParameter(
    name='eval-steps',
    default=5000,
    ptype=int,
)

examples = external_input(data_dir)
example_gen = CsvExampleGen(input=examples)

# Computes statistics over data for visualization and example validation.
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

# Generates schema based on statistics files.
schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    infer_feature_shape=False)

# Performs anomaly detection based on statistics and data schema.
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])

# Performs transformations and feature engineering in training and serving.
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=module_file)

# Update ai_platform_training_args if distributed training was enabled.
# Number of worker machines used in distributed training.
worker_count = data_types.RuntimeParameter(
    name='worker-count',
    default=2,
    ptype=int,
)

# Type of worker machines used in distributed training.
worker_type = data_types.RuntimeParameter(
    name='worker-type',
    default='standard',
    ptype=str,
)

#   if FLAGS.distributed_training:
ai_platform_training_args.update({
    # You can specify the machine types, the number of replicas for workers
    # and parameter servers.
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#ScaleTier
    'scaleTier': 'CUSTOM',
    'masterType': 'large_model',
    'workerType': worker_type,
    'parameterServerType': 'standard',
    'workerCount': worker_count,
    'parameterServerCount': 1
})

# Uses user-provided Python function that implements a model using TF-Learn
# to train a model on Google Cloud AI Platform.
trainer = Trainer(
    custom_executor_spec=executor_spec.ExecutorClassSpec(
        ai_platform_trainer_executor.Executor),
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],
    schema=schema_gen.outputs['schema'],
    transform_graph=transform.outputs['transform_graph'],
    train_args={'num_steps': train_steps},
    eval_args={'num_steps': eval_steps},
    custom_config={
        ai_platform_trainer_executor.TRAINING_ARGS_KEY:
            ai_platform_training_args
    })

# Get the latest blessed model for model validation.
model_resolver = ResolverNode(
    instance_name='latest_blessed_model_resolver',
    resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing))

# Uses TFMA to compute a evaluation statistics over features of a model and
# perform quality validation of a candidate model (compared to a baseline).
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(signature_name='eval')],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            thresholds={
                'binary_accuracy':
                    tfma.config.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.6}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10}))
            })
    ])

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    # Change threshold will be ignored if there is no baseline (first run).
    eval_config=eval_config)

# Checks whether the model passed the validation steps and pushes the model
# to  Google Cloud AI Platform if check passed.
pusher = Pusher(
    custom_executor_spec=executor_spec.ExecutorClassSpec(
        ai_platform_pusher_executor.Executor),
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    custom_config={
        ai_platform_pusher_executor.SERVING_ARGS_KEY: ai_platform_serving_args
    })

components = [
    example_gen, statistics_gen, schema_gen, example_validator, transform,
    trainer, model_resolver, evaluator, pusher
]

p = pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=components,
    beam_pipeline_args=beam_pipeline_args,
  )



# def init_beam_pipeline(components, pipeline_root:Text, direct_num_workers:int) -> pipeline.Pipeline:
    
#     absl.logging.info(f'Pipeline root set to: {pipeline_root}')
#     beam_arg = [
#         f'--direct_num_workers={direct_num_workers}',
#         f'--requirements_file={requirement_file}'# optional
#     ]

#     p = pipeline.Pipeline(pipeline_name=pipeline_name,
#                           pipeline_root=pipeline_root,
#                           components=components,
#                           enable_cache=False,
#                           metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
#                           beam_pipeline_args=beam_arg)
#     return p


if __name__ == '__main__':

    absl.logging.set_verbosity(absl.logging.INFO)
    # components = init_components(data_dir, module_file, serving_model_dir, 
    #                              training_steps=100, eval_steps=100)
    # direct_num_workers = int(os.cpu_count() / 2)
    # direct_num_workers = 1 if direct_num_workers < 1 else direct_num_workers
    # pipeline = init_beam_pipeline(components, pipeline_root, direct_num_workers)
    # BeamDagRunner().run(pipeline)


    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    # This pipeline automatically injects the Kubeflow TFX image if the
    # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
    # cli tool exports the environment variable to pass to the pipelines.
    tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', 'gcr.io/oreilly-book/ml-pipelines-tfx-custom:0.21.3')

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        # Specify custom docker image to use.
        tfx_image=tfx_image)

    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config,
        output_dir=output_dir, 
        output_filename=output_filename).run(p)
