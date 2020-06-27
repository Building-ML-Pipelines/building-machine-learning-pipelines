import os
import sys
from typing import Text

import absl
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

pipeline_name = "consumer_complaint_pipeline_beam"

# pipeline inputs
pipeline_dir = os.getcwd()
data_dir = os.path.join(pipeline_dir, "data")
module_file = os.path.join(pipeline_dir, "components", "module.py")
requirement_file = os.path.join(pipeline_dir, "requirements.txt")

# pipeline outputs
output_base = os.path.join(pipeline_dir, "output", pipeline_name)
serving_model_dir = os.path.join(output_base, pipeline_name)
pipeline_root = os.path.join(output_base, "pipeline_root")
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_beam_pipeline(
    components, pipeline_root: Text, direct_num_workers: int
) -> pipeline.Pipeline:

    absl.logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_arg = [
        f"--direct_num_workers={direct_num_workers}",
        f"--requirements_file={requirement_file}",  # optional
    ]

    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_arg,
    )
    return p


if __name__ == "__main__":

    absl.logging.set_verbosity(absl.logging.INFO)

    module_path = os.getcwd()
    if module_path not in sys.path:
        print(module_path)
        sys.path.append(module_path)

    from pipelines.base_pipeline import init_components

    components = init_components(
        data_dir,
        module_file,
        training_steps=50000,
        eval_steps=10000,
        serving_model_dir=serving_model_dir,
    )
    direct_num_workers = int(os.cpu_count() / 2)
    direct_num_workers = 1 if direct_num_workers < 1 else direct_num_workers
    pipeline = init_beam_pipeline(components, pipeline_root, direct_num_workers)
    BeamDagRunner().run(pipeline)
