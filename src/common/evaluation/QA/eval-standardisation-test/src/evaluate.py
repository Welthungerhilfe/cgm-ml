import os
import pickle
import random
from pathlib import Path

import glob2 as glob
import numpy as np
import pandas as pd
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from IPython.display import display
from tensorflow.keras.models import load_model

import utils
from constants import REPO_DIR
from qa_config import DATA_CONFIG, EVAL_CONFIG, MODEL_CONFIG, RESULT_CONFIG


class DataGenerator(tf.keras.utils.Sequence):
    '''
    Generator Class to create dataset in batches
    '''

    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size

    def __len__(self):
        length = int(len(self.X) / self.batch_size)
        if length * self.batch_size < len(self.X):
            length += 1
        return length

    def __getitem__(self, index):
        X = self.X[index * self.batch_size: (index + 1) * self.batch_size]
        return self.__getdepthmap__(X)

    def __getdepthmap__(self, depthmap_path_list):
        depthmaps = []
        for depthmap_path in depthmap_path_list:
            data, width, height, depthScale, _ = utils.load_depth(depthmap_path)
            depthmap, height, width = utils.prepare_depthmap(data, width, height, depthScale)
            depthmap = utils.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps_to_predict = tf.stack(depthmaps)
        return depthmaps_to_predict


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    '''
    Utility to load the depthmap pickle file
    '''
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = utils.preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH))
        targets = utils.preprocess_targets(targets, DATA_CONFIG.TARGET_INDEXES)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(DATA_CONFIG.TARGET_INDEXES,)))
    return depthmap, targets


def get_height_prediction(MODEL_PATH, dataset_evaluation):
    '''
    Perform the height prediction on the dataset
    Input:
        MODEL_PATH : Path of the trained model
        dataset_evaluation : dataset in which Evaluation
        need to performed
    '''
    model = load_model(MODEL_PATH, compile=False)
    predictions = model.predict(DataGenerator(depthmap_path_list, DATA_CONFIG.BATCH_SIZE))
    prediction_list = np.squeeze(predictions)
    return prediction_list


def prepare_depthmap_measure_table(depthmap_path_list, prediction_list):
    '''
    Prepares the measure table from the prediction by model
    on the depthmap
    Input:
        depthmap_path_list : List of Depthmaps paths of standardised dataset
        prediction_list : List of prediction on the above depthmap
    Output:
        depthmap_measure_table : Depthmap Measure Table for qrcode in the
        standardised dataset
    '''
    df = pd.DataFrame({'depthmap_path': depthmap_path_list, 'prediction': prediction_list})
    df['enumerator'] = df['depthmap_path'].apply(lambda x: x.split('/')[-3])
    df['qrcode'] = df['depthmap_path'].apply(lambda x: x.split('/')[-5])
    df['scantype'] = df['depthmap_path'].apply(lambda x: x.split('/')[-1].split('_')[-2])
    df['MeasureGroup'] = 'NaN'

    grp_by_col = ['enumerator', 'qrcode', 'scantype']
    measure_group_code = ['Height 1', 'Height 2']
    #Group for two measurement for TEM
    groups_for_two_measure = df.groupby(grp_by_col)

    if EVAL_CONFIG.DEBUG_LOG:
        display(groups_for_two_measure)

    for idx, (name, group) in enumerate(groups_for_two_measure):
        length = group.index.size

        #Check if qrcode contains contains multiple artifact or not
        if length > 1:
            measure_grp_one = group.index.values[: length // 2]
            measure_grp_two = group.index[length // 2:]

            df.loc[measure_grp_one, 'MeasureGroup'] = measure_group_code[0]
            df.loc[measure_grp_two, 'MeasureGroup'] = measure_group_code[1]

        # TODO: handle if qrcode contains single artifact
        # In current case it will not consider qrcode for
        # standardisation test

    df = df.dropna(axis=0)
    if EVAL_CONFIG.DEBUG_LOG:
        display(df.describe())
        display(df.head(4))

    # depthmap_measure_table contains measurement predicted by model
    depthmap_measure_table = pd.pivot_table(df, values='prediction', index='qrcode',
                                            columns=['enumerator', 'MeasureGroup'], aggfunc=np.mean)
    depthmap_measure_table = depthmap_measure_table.dropna(axis=1)

    #convert multi index column to single index column
    single_index_column = pd.Index([col[0] + '_' + col[1] for col in depthmap_measure_table.columns.tolist()])
    depthmap_measure_table.columns = single_index_column

    if EVAL_CONFIG.DEBUG_LOG:
        display(depthmap_measure_table)

    return depthmap_measure_table


def prepare_final_measure_table(excel_path, sheet_name, depthmap_measure_table):
    """
    Merge enumerator measure table and depthmap measure table
    and return final table
    Input:
        excel_path : Excel file path of the measure done by enumerator
        of Standardisation test
        sheet_name : Name of the sheet in the excel
        depthmap_measure_table : Measure table of the qrcode based on
        prediction from the model
    Output :
        final_measure_table : Final Measure table of the prediction done
        using model and enumerator
    """
    standardisation_df = pd.read_excel(excel_path, header=[0, 1], sheet_name=sheet_name)
    standardisation_df.drop(['ENUMERATOR NO.7', 'ENUMERATOR NO.8'], axis=1, inplace=True)

    if EVAL_CONFIG.DEBUG_LOG:
        display(standardisation_df)
        display(standardisation_df.columns)

    #Convert multiindex columns to single index columns
    single_index_col = pd.Index([col[0] + '_' + col[1] for col in standardisation_df.columns.tolist()])
    standardisation_df.columns = single_index_col

    #Rename column name to qrcode
    standardisation_df.rename(columns={'Unnamed: 1_level_0_QR Code': 'qrcode'}, inplace=True)

    #set index to qrcode column
    standardisation_df.set_index('qrcode', inplace=True)

    #drop unused columns
    standardisation_df.drop('Unnamed: 0_level_0_Child Number ', inplace=True, axis=1)
    if EVAL_CONFIG.DEBUG_LOG:
        display(standardisation_df)
        display("Index of dfs: ", standardisation_df.index)

    final_measure_table = pd.concat([depthmap_measure_table, standardisation_df], axis=1)

    if EVAL_CONFIG.DEBUG_LOG:
        display(final_measure_table)

    return final_measure_table


def prepare_and_save_tem_results(measure_table, save_path):
    '''
    Calculate TEM using the measure table for enumerator and trained model
    Input:
        measure_table : Final measure table containing measurement from
        enumerator and trained model
        save_path : Path to save the TEM results
    Output:
        result : TEM result for each measurer(enumerator and trained model)
    '''
    result_index = list(set([col.split('_')[0] for col in measure_table.columns]))
    result_col = ['TEM']
    result = pd.DataFrame(columns=result_col, index=result_index)

    for idx in result.index:
        heightOne = measure_table[idx + '_Height 1']
        heightTwo = measure_table[idx + '_Height 2']
        tem = utils.get_intra_TEM(heightOne, heightTwo)
        result.loc[idx, 'TEM'] = tem

    new_index = []
    for idx in result.index:
        if idx[:3] == 'cgm':
            new_index.append('Model_Measure_' + idx)
        else:
            new_index.append(idx)
    result.index = new_index

    if EVAL_CONFIG.DEBUG_LOG:
        display(result)

    result.to_csv(save_path)
    return result


if __name__ == "__main__":

    utils.setWidth(int(240 * 0.75))
    utils.setHeight(int(180 * 0.75))

    # Make experiment reproducible
    tf.random.set_seed(EVAL_CONFIG.SPLIT_SEED)
    random.seed(EVAL_CONFIG.SPLIT_SEED)

    # Get the current run.
    run = Run.get_context()

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if(run.id.startswith("OfflineRun")):
        print("Running in offline mode...")

        # Access workspace.
        print("Accessing workspace...")
        workspace = Workspace.from_config()
        experiment = Experiment(workspace, EVAL_CONFIG.EXPERIMENT_NAME)
        run = experiment.start_logging(outputs=None, snapshot_directory=None)

        # Get dataset.
        print("Accessing dataset...")
        dataset_name = DATA_CONFIG.NAME
        dataset_path = str(REPO_DIR / "data" / dataset_name)
        print("Dataset Path: ", dataset_path)
        if not os.path.exists(dataset_path):
            dataset = workspace.datasets[dataset_name]
            dataset.download(target_path=dataset_path, overwrite=False)

    # Online run. Use dataset provided by training notebook.
    else:
        print("Running in online mode...")
        experiment = run.experiment
        workspace = experiment.workspace
        dataset_path = run.input_datasets["dataset"]

    qrcode_path = os.path.join(dataset_path, "qrcode")
    print("QRcode path:", qrcode_path)
    print("Getting Depthmap paths...")
    depthmap_path_list = glob.glob(os.path.join(qrcode_path, "*/measure/*/depth/*"))
    assert len(depthmap_path_list) != 0

    if EVAL_CONFIG.DEBUG_RUN and len(depthmap_path_list) > EVAL_CONFIG.DEBUG_NUMBER_OF_DEPTHMAP:
        depthmap_path_list = depthmap_path_list[:EVAL_CONFIG.DEBUG_NUMBER_OF_DEPTHMAP]
        print("Executing on {} qrcodes for FAST RUN".format(EVAL_CONFIG.DEBUG_NUMBER_OF_DEPTHMAP))

    print("Paths for Depth map Evaluation:")
    print("\t" + "\n\t".join(depthmap_path_list))
    print("Using {} artifact files for evaluation.".format(len(depthmap_path_list)))

    # Get the prediction on the artifact
    if MODEL_CONFIG.NAME.endswith(".h5"):
        model_path = MODEL_CONFIG.NAME
    elif MODEL_CONFIG.NAME.endswith(".ckpt"):
        model_path = f"{MODEL_CONFIG.INPUT_LOCATION}/{MODEL_CONFIG.NAME}"
    else:
        raise NameError(f"{MODEL_CONFIG.NAME}'s path extension not supported")
    prediction_list = get_height_prediction(model_path, depthmap_path_list)

    depthmap_measure_table = prepare_depthmap_measure_table(depthmap_path_list, prediction_list)

    data_path = Path(dataset_path)
    excel_path = data_path / DATA_CONFIG.EXCEL_NAME
    sheet_name = DATA_CONFIG.SHEET_NAME

    final_measure_table = prepare_final_measure_table(excel_path, sheet_name, depthmap_measure_table)
    result = prepare_and_save_tem_results(final_measure_table, RESULT_CONFIG.SAVE_TEM_PATH)

    # Done.
    run.complete()
