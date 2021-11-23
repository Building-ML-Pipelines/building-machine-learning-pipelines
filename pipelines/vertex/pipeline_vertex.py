import os
import sys

from absl import logging
from tfx import v1 as tfx
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner


pipeline_name = "consumer_complaint_pipeline_cloud_ai_to_cloud_bucket"

# temp yaml file for Kubeflow Pipelines
output_filename = f"{pipeline_name}.yaml"
output_dir = os.path.join(
    os.getcwd(), "pipelines", "gcp_cloud_ai", "argo_pipeline_files"
)

# Directory and data locations (uses Google Cloud Storage).
input_bucket = "gs://consumer_complaint_gcp_cloud_ai"
output_bucket = "gs://consumer_complaint_gcp_cloud_ai"
data_dir = os.path.join(input_bucket, "data")

tfx_root = os.path.join(output_bucket, "tfx_pipeline")
pipeline_root = os.path.join(tfx_root, pipeline_name)
ai_platform_distributed_training = False
serving_model_dir = os.path.join(output_bucket, "serving_model_dir")

# Google Cloud Platform project id to use when deploying this pipeline.
project_id = "~~oreilly-book~~"  # <--- needs update by the user

module_file = os.path.join(input_bucket, "components", "module.py")

gcp_region = "us-central1"

use_gpu = True

vertex_training_args = {
    "project": project_id,
    "worker_pool_specs": [
        {
            "machine_spec": {
                "machine_type": "n1-highmem-8",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "gcr.io/tfx-oss-public/tfx:{}".format(tfx.__version__),
            },
        }
    ],
}

if use_gpu:
    vertex_training_args["worker_pool_specs"][0]["machine_spec"].update(
        {"accelerator_type": "NVIDIA_TESLA_K80", "accelerator_count": 1}
    )

vertex_training_custom_config = {
    tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY: True,
    tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY: gcp_region,
    tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_training_args,
    "use_gpu": use_gpu,
}


vertex_serving_spec = {
    "project_id": project_id,
    "endpoint_name": "consumer_complaint",
    "deployed_model_display_name": "consumer_complaint",
    "machine_type": "n1-standard-2",
    "min_replica_count": 1,
    "max_replica_count": 2,
    "metadata": (("model_name", "consumer_complaint"),),
}

vertex_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest"

vertex_serving_args = {
    tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
    tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: gcp_region,
    tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: vertex_container_image_uri,
    tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: vertex_serving_spec,
}

beam_pipeline_args = [
    "--runner=DataflowRunner",
    "--experiments=shuffle_mode=auto",
    "--project=" + project_id,
    "--temp_location=" + os.path.join(output_bucket, "tmp"),
    "--region=" + gcp_region,
    "--disk_size_gb=50",
    "--machine_type=e2-standard-8",
    "--experiments=use_runner_v2",
]


if __name__ == "__main__":

    logging.set_verbosity(logging.INFO)

    module_path = os.getcwd()
    if module_path not in sys.path:
        sys.path.append(module_path)

    from pipelines.base_pipeline import init_components

    components = init_components(
        data_dir,
        module_file,
        vertex_training_custom_config=vertex_training_custom_config,
        serving_model_dir=serving_model_dir,
        # ai_platform_serving_args=ai_platform_serving_args
    )

    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args,
    )

    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    # This pipeline automatically injects the Kubeflow TFX image if the
    # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
    # cli tool exports the environment variable to pass to the pipelines.
    tfx_image = os.environ.get(
        "KUBEFLOW_TFX_IMAGE",
        "gcr.io/oreilly-book/ml-pipelines-tfx-custom:latest",
    )

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        # Specify custom docker image to use.
        tfx_image=tfx_image,
    )

    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config,
        output_dir=output_dir,
        output_filename=output_filename,
    ).run(p)
