#import glob as glob
import glob2 as glob
import pandas as pd
import numpy as np
from pathlib import Path
import xlrd
import math
import utils
from skimage.transform import resize

import os
import pickle
import random
import matplotlib.pyplot as plt
from IPython.display import display

import tensorflow as tf
from azureml.core import Experiment, Workspace, Dataset
from azureml.core.run import Run

from test_config import MODEL_CONFIG, EVAL_CONFIG, DATA_CONFIG, RESULT_CONFIG
from constants import REPO_DIR
from tensorflow.keras.models import load_model


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size

    def __len__(self):
        l = int(len(self.X) / self.batch_size)
        if l*self.batch_size < len(self.X):
            l += 1
        return l

    def __getitem__(self, index):
        X = self.X[index*self.batch_size:(index+1)*self.batch_size]
        return self.__getdepthmap__(X)
    
    def __getdepthmap__(self, depthmap_path_list):
        depthmaps = []
        #target_heights = []
        for depthmap_path in depthmap_path_list:
            data, width, height, depthScale, _ = utils.load_depth(depthmap_path)
            depthmap,height, width = utils.prepare_depthmap(data, width, height, depthScale)
            #print(height, width)
            depthmap = utils.preprocess(depthmap)
            #print(depthmap.shape)
            depthmaps.append(depthmap)
                
        depthmaps_to_predict = tf.stack(depthmaps)
        #depthmaps = np.array(depthmaps)
        return depthmaps_to_predict


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
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

'''
def get_depthmaps(paths):
    depthmaps = []
    for path in paths:
        data, width, height, depthScale, maxConfidence = utils.load_depth(path)
        depthmap,height, width = utils.prepare_depthmap(data, width, height, depthScale)
        #print(height, width)
        depthmap = utils.preprocess(depthmap)
        #print(depthmap.shape)
        depthmaps.append(depthmap)
    depthmaps = np.array(depthmaps)
    return depthmaps
'''


def get_height_prediction(MODEL_PATH, dataset_evaluation):
    model = load_model(MODEL_PATH)
    predictions = model.predict(DataGenerator(depthmap_path_list, DATA_CONFIG.BATCH_SIZE))
    prediction_list = np.squeeze(predictions)
    return prediction_list


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
        print("A: ", dataset_path)
        if not os.path.exists(dataset_path):
            dataset = workspace.datasets[dataset_name]
            #Download depthmap and QR code
            #dataset = Dataset.get_by_name(workspace, name='standardization_test_01', version="1")
            #dataset.download(target_path=dataset_path, overwrite=False)

            #Download Standadisation Excel
            #dataset = Dataset.get_by_name(workspace, name='standardization_test_01')
            dataset.download(target_path=dataset_path, overwrite=False)

    # Online run. Use dataset provided by training notebook.
    else:
        print("Running in online mode...")
        # TODO : Need to verify the online part
        experiment = run.experiment
        workspace = experiment.workspace
        dataset_path = run.input_datasets["dataset"]

    print("B: ", dataset_path)
    print("Content of Dataset: ", os.listdir(dataset_path))
    qrcode_path = os.path.join(dataset_path, "qrcode")
    print("QRcode path:", qrcode_path)
    print(glob.glob(os.path.join(qrcode_path, "*"))) # Debug
    print("Getting Depthmap paths...")
    depthmap_path_list = glob.glob(os.path.join(qrcode_path, "*/measure/*/depth/*"))
    print("depthmap_path_list: ", len(depthmap_path_list))
    assert len(depthmap_path_list) != 0

    if EVAL_CONFIG.DEBUG_RUN and len(depthmap_path_list) > EVAL_CONFIG.DEBUG_NUMBER_OF_DEPTHMAP:
        depthmap_path_list = depthmap_path_list[:EVAL_CONFIG.DEBUG_NUMBER_OF_DEPTHMAP]
        print("Executing on {} qrcodes for FAST RUN".format(EVAL_CONFIG.DEBUG_NUMBER_OF_DEPTHMAP))


    print("Paths for evaluation:")
    print("\t" + "\n\t".join(depthmap_path_list))

    print("Depth Map Length : ", len(depthmap_path_list))


    print("Using {} artifact files for evaluation.".format(len(depthmap_path_list)))


    prediction_list = get_height_prediction(MODEL_CONFIG.NAME, depthmap_path_list)

    df = pd.DataFrame({'depthmap_path': depthmap_path_list, 'prediction': prediction_list})
    df['enumerator'] = df['depthmap_path'].apply(lambda x : x.split('/')[-3])
    df['qrcode'] = df['depthmap_path'].apply(lambda x : x.split('/')[-5])
    df['scantype'] = df['depthmap_path'].apply(lambda x : x.split('/')[-1].split('_')[-2])
    df['MeasureGroup'] = 'NaN'
    display(df.head(4))
    
    grp_col_list = ['enumerator', 'qrcode', 'scantype']
    MeasureGroup_code = ['Height 1', 'Height 2']
    abc = df.groupby(grp_col_list)
    abc


    for idx, (name, group) in enumerate(abc):
        #if idx == 4:
        #    break
        #print(name)
        #print(group.index)
        #df[group.index]
        #print(group.index.values)
        length = group.index.size

        #Check if qrcode contains contains multiple artifact or not
        if length > 1:
            measure_grp_one = group.index.values[: length//2]
            measure_grp_two = group.index[length//2:]

            df.loc[measure_grp_one, 'MeasureGroup'] = MeasureGroup_code[0]
            df.loc[measure_grp_two, 'MeasureGroup'] = MeasureGroup_code[1]
            

    display(df.head(10))
    display(df.describe())

    hello = df.copy()
    hello = hello.dropna(axis = 0)
    display(hello.describe())
    display(hello.head(4))

    cde = pd.pivot_table(hello, values = 'prediction', index = 'qrcode', 
                columns= ['enumerator', 'MeasureGroup'], aggfunc = np.mean)
    cde = cde.dropna(axis = 1)
    display(cde)

    ind = pd.Index([e[0] + '_'+e[1] for e in cde.columns.tolist()])
    cde.columns = ind
    display(cde)

    data_path = Path(dataset_path)
    #excel_path
    #excel_path = data_path / 'Standardization-test-October.xlsx'
    excel_path = data_path / 'Standardization test October.xlsx'
    sheet_name = 'DRS'
    
    dfs = pd.read_excel(excel_path, header=[0, 1], sheet_name=sheet_name)
    dfs.drop(['ENUMERATOR NO.7', 'ENUMERATOR NO.8'], axis = 1, inplace = True)
    display(dfs)
    display(dfs.columns)
    ind = pd.Index([e[0] + '_'+e[1] for e in dfs.columns.tolist()])
    display(ind)
    dfs.columns = ind
    dfs.rename(columns = {'Unnamed: 1_level_0_QR Code':'qrcode'}, inplace = True) 
    dfs.set_index('qrcode', inplace = True)
    #dfs.drop['Unnamed: 0_level_0_Child Number']
    dfs.drop('Unnamed: 0_level_0_Child Number ', inplace=True, axis=1) 
    display(dfs)
    display("Index of dfs: ", dfs.index)

    df5 = pd.concat([cde, dfs], axis = 1)
    display(df5)

    heightOne = df5['ENUMERATOR NO.6_Height 1']
    heightTwo = df5['ENUMERATOR NO.6_Height 2']
    tem = utils.get_intra_TEM(heightOne, heightTwo)
    print(tem)
    print(utils.get_meaure_category(tem, 'HEIGHT'))

    heightOne = df5['cgm01drs_Height 1']
    heightTwo = df5['cgm01drs_Height 2']
    tem = utils.get_intra_TEM(heightOne, heightTwo)
    print(tem)
    print(utils.get_meaure_category(tem, 'HEIGHT'))


    result_index = list(set([col.split('_')[0] for col in df5.columns]))
    result_col = ['TEM']
    result = pd.DataFrame(columns = result_col, index = result_index)
    display(result)

    for idx in result.index:
        #print(idx)
        heightOne = df5[idx + '_Height 1']
        heightTwo = df5[idx + '_Height 2']
        tem = utils.get_intra_TEM(heightOne, heightTwo)
        #print(tem)
        #print(utils.get_meaure_category(tem, 'HEIGHT'))
        result.loc[idx, 'TEM'] = tem
    
    display(result)
    #result
    col = []
    for idx in result.index:
        if result.loc[idx, 'TEM'] < 0.2:
            col.append('blue')
        elif result.loc[idx, 'TEM'] >= 0.2:
            col.append('red')

    plt.bar(list(result.index), list(result.TEM), color = col)
    plt.xticks(rotation='vertical')

    #print("Saving the results")
    #utils.calculate_and_save_results(MAE, EVAL_CONFIG.NAME, RESULT_CONFIG.SAVE_PATH)

    # Done.
    run.complete()
