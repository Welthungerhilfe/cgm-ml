from azureml.core import Workspace, Experiment, Run
from config import CONFIG

ws = Workspace.from_config()
exp = Experiment(workspace=ws, name=CONFIG.EXPERIMENT_NAME)
run = Run(exp, CONFIG.RUN_ID)
model = run.register_model(model_name=CONFIG.MODEL_NAME,
                           model_path='outputs')
print('Model register successfully')
