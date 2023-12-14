from pipelines.deployment_pipeline import continuous_deployment_pipeline
from pipelines.deployment_pipeline import deployment_pipeline, inference_pipeline
import click

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy and predict'

@click.command()
@click.option(
    '--config',
    '-c',
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help=''
)

@click.option(
    '--min-accuracy',
    delault=0.92,
    help=''
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = mlflow_model_deployer_component.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
        )
        
    if predict:
        inference_pipeline()

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_service[0])
        if service.is_running:
            print('The service is running')
        elif service.is_failed:
            print('An Error has occured')
    else:
        print('')