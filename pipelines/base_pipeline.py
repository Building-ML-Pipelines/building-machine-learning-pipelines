import os

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2


TRAIN_STEPS = 50000
EVAL_STEPS = 10000


def init_components(
    data_dir,
    module_file,
    training_steps=TRAIN_STEPS,
    eval_steps=EVAL_STEPS,
    serving_model_dir=None,
    ai_platform_training_args=None,
    ai_platform_serving_args=None,
):

    if serving_model_dir and ai_platform_serving_args:
        raise NotImplementedError(
            "Can't set ai_platform_serving_args and serving_model_dir at "
            "the same time. Choose one deployment option."
        )

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=9),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )

    example_gen = tfx.components.CsvExampleGen(
        input_base=os.path.join(os.getcwd(), data_dir), output_config=output
    )

    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )

    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )

    training_kwargs = {
        "module_file": module_file,
        "examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": trainer_pb2.TrainArgs(num_steps=training_steps),
        "eval_args": trainer_pb2.EvalArgs(num_steps=eval_steps),
    }

    # if ai_platform_training_args:
    #     from tfx.extensions.google_cloud_ai_platform.trainer import (
    #         executor as aip_trainer_executor,
    #     )

    #     training_kwargs.update(
    #         {
    #             "custom_executor_spec": executor_spec.ExecutorClassSpec(
    #                 aip_trainer_executor.GenericExecutor
    #             ),
    #             "custom_config": {
    #                 aip_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args  # noqa
    #             },
    #         }
    #     )

    trainer = tfx.components.Trainer(**training_kwargs)

    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing),
    )

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_default",
                label_key="consumer_disputed",
                # preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[tfma.SlicingSpec(), tfma.SlicingSpec(feature_keys=["product"])],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.65}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                ],
            )
        ],
    )

    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    pusher_kwargs = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
    }

    # if ai_platform_serving_args:
    #     from tfx.extensions.google_cloud_ai_platform.pusher import (
    #         executor as aip_pusher_executor,
    #     )

    #     pusher_kwargs.update(
    #         {
    #             "custom_executor_spec": executor_spec.ExecutorClassSpec(
    #                 aip_pusher_executor.Executor
    #             ),
    #             "custom_config": {
    #                 aip_pusher_executor.SERVING_ARGS_KEY: ai_platform_serving_args  # noqa
    #             },
    #         }
    #     )
    if serving_model_dir:
        pusher_kwargs.update(
            {
                "push_destination": pusher_pb2.PushDestination(
                    filesystem=pusher_pb2.PushDestination.Filesystem(
                        base_directory=serving_model_dir
                    )
                )
            }
        )
    else:
        raise NotImplementedError(
            "Provide ai_platform_serving_args or serving_model_dir."
        )

    pusher = tfx.components.Pusher(**pusher_kwargs)

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
