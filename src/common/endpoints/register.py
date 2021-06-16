from azureml.core import Experiment, Run, Workspace
from config_deepensemble_1 import CONFIG

if __name__ == "__main__":

    workspace = Workspace.from_config()
    experiment = Experiment(workspace=workspace, name=CONFIG.EXPERIMENT_NAME)
    run = Run(experiment, CONFIG.RUN_ID)
    model = run.register_model(model_name=CONFIG.MODEL_NAME,
                               model_path='outputs')
    print('Model register successfully')
