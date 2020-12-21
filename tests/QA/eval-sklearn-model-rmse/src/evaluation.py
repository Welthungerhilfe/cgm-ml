from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core import Run, Experiment
#import pickle

import math
import joblib

run = Run.get_context()

print("Running in online mode...")
experiment = run.experiment
workspace = experiment.workspace
dataset_ref = run.input_datasets["dataset"]

x_df = dataset_ref.to_pandas_dataframe()[['weight', 'height', 'muac']].dropna()
y_df = x_df.pop("muac")

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=66)

filename = 'model_alpha_1.0.pkl'

run = experiment.start_logging()


# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, y_test)
y_pred = loaded_model.predict(X=X_test)
rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
print("rmse", rmse)
run.log("rmse", rmse)

#print(result)


#run.log("Result", result)

#joblib.dump(value=model, filename=filename)
#run.upload_file(name=model_name, path_or_stream=filename)
run.complete()
