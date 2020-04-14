import os
from typing import List, Text

import absl
import tensorflow_model_analysis as tfma
from tfx.components import (CsvExampleGen, Evaluator, ExampleValidator, Pusher,
                            ResolverNode, SchemaGen, StatisticsGen, Trainer,
                            Transform)
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.utils.dsl_utils import external_input


TRAIN_STEPS = 100
EVAL_STEPS = 100

pipeline_name = 'consumer_complaint_pipeline_beam'
pipeline_dir = os.getcwd()
data_dir = os.path.join(pipeline_dir, 'data')
module_file = os.path.join(pipeline_dir, 'components/module.py')
serving_model_dir = os.path.join(pipeline_dir, 'serving_model', pipeline_name)

pipeline_root = os.path.join(pipeline_dir, 'tfx-2', 'beam', pipeline_name)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')
requirement_file = os.path.join(pipeline_dir, 'requirements.txt')


def init_components():
    examples = external_input(data_dir)
    example_gen = CsvExampleGen(input=examples)

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])
    
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)

    trainer = Trainer(
        module_file=module_file,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_STEPS))

    model_resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing))

    eval_config=tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='consumer_disputed')],
        slicing_specs=[tfma.SlicingSpec(), tfma.SlicingSpec(feature_keys=['product'])],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='ExampleCount'),
                ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    return components


def init_pipeline(components, pipeline_root:Text, direct_num_workers:int) -> pipeline.Pipeline:
    
    absl.logging.info(f'Pipeline root set to: {pipeline_root}')
    beam_arg = [
        f'--direct_num_workers={direct_num_workers}',
        f'--requirements_file={requirement_file}'
    ]
    p = pipeline.Pipeline(pipeline_name=pipeline_name,
                          pipeline_root=pipeline_root,
                          components=components,
                          enable_cache=True,
                          metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
                          beam_pipeline_args=beam_arg)
    return p
    

if __name__ == '__main__':

    absl.logging.set_verbosity(absl.logging.INFO)
    components = init_components()
    pipeline = init_pipeline(components, pipeline_root, 0)
    BeamDagRunner().run(pipeline)
