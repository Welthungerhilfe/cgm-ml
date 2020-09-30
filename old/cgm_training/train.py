#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
sys.path.insert(0, "..")


import warnings
warnings.filterwarnings("ignore")
from cgmcore import modelutils
from cgmcore import utils
import numpy as np
from tensorflow.keras import callbacks, optimizers, models, layers
import tensorflow as tf
import pprint
import os
from cgmcore.preprocesseddatagenerator import get_dataset_path, create_datagenerator_from_parameters
import random
import qrcodes
from cgmcore.utils import create_training_tasks
import multiprocessing
from bunch import Bunch
import json
import pprint
import argparse
import shutil
import logging


# Set tensorflow log level.
tf.get_logger().setLevel('WARNING')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Training on GPU")
    parser.add_argument(
        "-config_file",
        action="store",
        dest="config_file",
        type=str,
        required=True,
        help="config file path"
    )
    parser.add_argument(
        "-use_multi_gpu",
        action="store_true",
        dest="use_multi_gpu",
        help="set the training on multiple gpus")
    parser.add_argument(
        "-resume_training",
        action="store_true",
        dest="resume_training",
        help="resumes a previous training")
    arguments = parser.parse_args()
    
    
    # Loading the config file.
    config = json.load(open(arguments.config_file, "r"))
    config = Bunch({ key: Bunch(value) for key, value in config.items()})

    # Create logger.
    logger = logging.getLogger("train.py")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(config.global_parameters.output_path, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Starting training job...")

    # Prepare results.
    results = Bunch()

    # Check if there is a GPU.
    if len(utils.get_available_gpus()) == 0:
        logger.warning("WARNING! No GPU available!")

    # Create datagenerator.
    datagenerator_instance = create_datagenerator_from_parameters(
        config.datagenerator_parameters.dataset_path, 
        config.datagenerator_parameters
    )

    # Do a test-validation split.
    qrcodes = datagenerator_instance.qrcodes[:]
    randomizer = random.Random(config.datagenerator_parameters.random_seed)
    randomizer.shuffle(qrcodes)
    split_index = int(0.8 * len(qrcodes))
    qrcodes_train = sorted(qrcodes[:split_index])
    qrcodes_validate = sorted(qrcodes[split_index:])
    del qrcodes
    results.qrcodes_train = qrcodes_train
    results.qrcodes_validate = qrcodes_validate

    # Create python generators.
    workers = 4
    generator_train = datagenerator_instance.generate(
        size=config.training_parameters.batch_size, 
        qrcodes_to_use=qrcodes_train, 
        workers=workers
    )
    generator_validate = datagenerator_instance.generate(
        size=config.training_parameters.batch_size,
        qrcodes_to_use=qrcodes_validate,
        workers=workers)

    # Output path. Ensure its existence.
    if os.path.exists(config.global_parameters.output_path) == False:
        os.makedirs(config.global_parameters.output_path)
    logger.info("Using output path:", config.global_parameters.output_path)

    # Copy config file.
    shutil.copy2(arguments.config_file, config.global_parameters.output_path)
    
    # Create the model path.
    model_path = os.path.join(config.global_parameters.output_path, "model.h5")
    
    # TODO
    assert config.model_parameters.type == "pointnet"
    
    # Resume training.
    if arguments.resume_training == True:
        if os.path.exists(model_path) == False:
            logger.error("Model does not exist. Cannot resume!")
            exit(0)
        model = tf.keras.models.load_model(model_path)
        logger.info("Loaded model from {}.".format(config.model_path))
    
    # Start from scratch.
    else:
        model = modelutils.create_point_net(
            config.model_parameters.input_shape, 
            config.model_parameters.output_size, 
            config.model_parameters.hidden_sizes
        )
        logger.info("Created new model.")
    model.summary()
    
    # Compile model.
    if config.model_parameters.optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(
            learning_rate=config.model_parameters.learning_rate
        )
    elif config.model_parameters.optimizer == "adam":
        optimizer = optimizers.Adam(
            learning_rate=config.model_parameters.learning_rate, 
            beta_1=config.model_parameters.beta_1, 
            beta_2=config.model_parameters.beta_2, 
            amsgrad=config.model_parameters.amsgrad
        )
    else:
        raise Exception("Unexpected optimizer {}".format(config.model_parameters.optimizer))


        
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    # Do training on multiple GPUs.
    original_model = model
    if arguments.use_multi_gpu == True:
        model = tf.keras.utils.multi_gpu_model(model, gpus=2)
    
    # Create the callbacks.
    callbacks = []
    
    # Logging training progress with tensorboard.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=config.global_parameters.output_path, 
        histogram_freq=0, 
        batch_size=32, 
        write_graph=True, 
        write_grads=False, 
        write_images=True, 
        embeddings_freq=0, 
        embeddings_layer_names=None, 
        embeddings_metadata=None, 
        embeddings_data=None, 
        update_freq="epoch"
    )
    callbacks.append(tensorboard_callback)

    # Early stopping.
    if config.training_parameters.use_early_stopping == True:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=config.training_parameters.early_stopping_threshold,
            patience=5,
            verbose=1
        )
        callbacks.append(early_stopping_callback)

    # Model checkpoint.    
    val_loss_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.global_parameters.output_path, "val_loss_{val_loss:.2f}_at_epoche_{epoch:2d}.hdf5"), 
        monitor="val_loss", 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False, 
        mode="auto", 
        save_freq="epoch")
    callbacks.append(val_loss_callback)    
    
    # Start training.
    results.training_begin = utils.get_datetime_string()
    try:
        # Train the model.
        model.fit_generator(
            generator_train,
            steps_per_epoch=config.training_parameters.steps_per_epoch,
            epochs=config.training_parameters.epochs,
            validation_data=generator_validate,
            validation_steps=config.training_parameters.validation_steps,
            use_multiprocessing=False,
            workers=0,
            callbacks=callbacks
            )
    except KeyboardInterrupt:
        logger.info("Gracefully stopping training...")
        datagenerator_instance.finish()
        results.interrupted_by_user = True
           
    # Training ended.
    results.training_end = utils.get_datetime_string()

    # Save the model. Make sure that it is the original model.
    original_model.save(model_path)
    
    # Store the history.
    results.model_history = model.history.history

    # Write the results.
    results_name = "results.json"
    results_path = os.path.join(config.global_parameters.output_path, results_name)
    json.dump(results, open(results_path, "w"), indent=4, sort_keys=True)

    
if __name__ == "__main__":
    main()
