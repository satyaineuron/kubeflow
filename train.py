import os

from kfp import Client
from kfp.components import load_component_from_file
from kfp.dsl import ContainerOp, pipeline
from kubernetes.client.models import V1EnvVar

from ecom.api.auth import get_istio_auth_session

KUBEFLOW_ENDPOINT = os.environ["KUBEFLOW_ENDPOINT"]
KUBEFLOW_USERNAME = os.environ["KUBEFLOW_USERNAME"]
KUBEFLOW_PASSWORD = os.environ["KUBEFLOW_PASSWORD"]

auth_session = get_istio_auth_session(
    url=KUBEFLOW_ENDPOINT, username=KUBEFLOW_USERNAME, password=KUBEFLOW_PASSWORD
)

data_ingestion = load_component_from_file("kfp_components/data_ingestion.yaml")

model_training = load_component_from_file("kfp_components/model_training.yaml")


@pipeline(name="Train Pipeline")
def train_pipeline():
    task_1: ContainerOp = data_ingestion()

    task_2: ContainerOp = model_training()

    task_2.after(task_1).container.set_image_pull_policy(image_pull_policy="Always")


if __name__ == "__main__":
    client = Client(
        host=f"{KUBEFLOW_ENDPOINT}/pipeline", cookies=auth_session["session_cookie"]
    )

    client.create_run_from_pipeline_func(
        pipeline_func=train_pipeline,
        arguments={},
        namespace="kubeflow-user-example-com",
        experiment_name="imdb-exp",
        service_account="kube-imdb-sa",
        enable_caching=False,
    ).wait_for_run_completion()
