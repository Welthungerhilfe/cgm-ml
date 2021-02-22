import os
from pathlib import Path

from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras import models

from config import CONFIG
from constants import MODEL_CKPT_FILENAME
from tmp_model_util.utils import create_base_cnn


def get_base_model(workspace: Workspace, data_dir: Path) -> models.Sequential:
    if CONFIG.PRETRAINED_RUN:
        model_fpath = data_dir / "pretrained" / CONFIG.PRETRAINED_RUN
        if not os.path.exists(model_fpath):
            download_pretrained_model(workspace, model_fpath)
        print(f"Loading pretrained model from {model_fpath}")
        base_model = load_base_cgm_model(model_fpath, should_freeze=CONFIG.SHOULD_FREEZE_BASE)
    else:
        input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1)
        base_model = create_base_cnn(input_shape, dropout=CONFIG.USE_DROPOUT)  # output_shape: (128,)
    return base_model


def download_model(ws, experiment_name, run_id, input_location, output_location):
    """Download the pretrained model

    Args:
         ws: workspace to access the experiment
         experiment_name: Name of the experiment in which model is saved
         run_id: Run Id of the experiment in which model is pre-trained
         input_location: Input location in a RUN Id
         output_location: Location for saving the model
    """
    experiment = Experiment(workspace=ws, name=experiment_name)
    # Download the model on which evaluation need to be done
    run = Run(experiment, run_id=run_id)
    if input_location.endswith(".h5"):
        run.download_file(input_location, output_location)
    elif input_location.endswith(".ckpt"):
        run.download_files(prefix=input_location, output_directory=output_location)
    else:
        raise NameError(f"{input_location}'s path extension not supported")
    print("Successfully downloaded model")


def download_pretrained_model(workspace: Workspace, output_model_fpath: str):
    print(f"Downloading pretrained model from {CONFIG.PRETRAINED_RUN}")
    download_model(ws=workspace,
                   experiment_name=CONFIG.PRETRAINED_EXPERIMENT,
                   run_id=CONFIG.PRETRAINED_RUN,
                   input_location=f"outputs/{MODEL_CKPT_FILENAME}",
                   output_location=output_model_fpath)


def load_base_cgm_model(model_fpath: str, should_freeze: bool = False) -> models.Sequential:
    # load model
    loaded_model = models.load_model(
        str(Path(model_fpath) / "outputs" / MODEL_CKPT_FILENAME)
    )

    # cut off last layer (https://stackoverflow.com/a/59304656/5497962)
    beheaded_model = models.Sequential(name="base_model_beheaded")
    for layer in loaded_model.layers[:-1]:
        beheaded_model.add(layer)

    if should_freeze:
        for layer in beheaded_model._layers:
            layer.trainable = False

    return beheaded_model
