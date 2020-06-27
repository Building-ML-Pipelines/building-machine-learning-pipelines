import os
import sys

import absl
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
project_id = "oreilly-book"  # <--- needs update by the user

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run
# successfully. Copy this from the current directory to a GCS bucket and update
# the location below.
module_file = os.path.join(input_bucket, "components", "module.py")

# Region to use for Dataflow jobs and AI Platform jobs.
#   Dataflow:
#       https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform:
#       https://cloud.google.com/ml-engine/docs/tensorflow/regions
gcp_region = "us-central1"

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google
# Cloud AI Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
ai_platform_training_args = {
    "project": project_id,
    "region": gcp_region,
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will
    # use a public container image matching the installed version of TFX.
    "masterConfig": {
        "imageUri": "gcr.io/oreilly-book/ml-pipelines-tfx-custom:0.22.0"
    },
    # important: Note that if you do specify a custom container, ensure the
    # entrypoint calls into TFX's run_executor script
    # (tfx/scripts/run_executor.py)
}

if ai_platform_distributed_training:

    # Update ai_platform_training_args if distributed training was enabled.
    # Number of worker machines used in distributed training.
    from tfx.orchestration import data_types

    worker_count = data_types.RuntimeParameter(
        name="worker-count", default=4, ptype=int,
    )

    # Type of worker machines used in distributed training.
    worker_type = data_types.RuntimeParameter(
        name="worker-type", default="standard", ptype=str,
    )

    ai_platform_training_args.update(
        {
            # You can specify the machine types, the number of replicas
            # for workers and parameter servers.
            # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#ScaleTier
            "scaleTier": "CUSTOM",
            "masterType": "large_model",
            "workerType": worker_type,
            "parameterServerType": "standard",
            "workerCount": worker_count,
            "parameterServerCount": 1,
        }
    )

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google
# Cloud AI Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
ai_platform_serving_args = {
    "model_name": "consumer_complaint",
    "project_id": project_id,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    "regions": [gcp_region],
}

beam_pipeline_args = [
    "--runner=DataflowRunner",
    "--experiments=shuffle_mode=auto",
    "--project=" + project_id,
    "--temp_location=" + os.path.join(output_bucket, "tmp"),
    "--region=" + gcp_region,
    "--disk_size_gb=50",
]


if __name__ == "__main__":

    absl.logging.set_verbosity(absl.logging.INFO)

    module_path = os.getcwd()
    if module_path not in sys.path:
        sys.path.append(module_path)

    from pipelines.base_pipeline import init_components

    components = init_components(
        data_dir,
        module_file,
        ai_platform_training_args=ai_platform_training_args,
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
        "gcr.io/oreilly-book/ml-pipelines-tfx-custom:0.21.4",
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
