import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core import ScriptRunConfig

ws = Workspace.from_config()
exp = Experiment(workspace=ws, name="q1-ensemble-warmup")
print(exp)
run = Run(exp, 'q1-ensemble-warmup_1620889029_86f10712')
model = run.register_model(model_name='Deepensemble',
                           model_path='outputs/best_model.ckpt')
