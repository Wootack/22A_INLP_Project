from sys import argv
from os.path import exists
from os import listdir
import pickle

from configurations import DF_VERSION

if len(argv) == 1:
    object_function = 'classifier'
else:
    object_function = argv[1]
train_df, test_df, full_sensor_list, df_status = [None] * 4
# if object_function != 'preprocess' and object_function != 'debugg':
if object_function != 'preprocess':
    if DF_VERSION == 'Recent':
        PREPROCESSED_FILEPATH = './cache/preprocessed'
        latest_pickle = sorted(listdir(PREPROCESSED_FILEPATH))[-2]
        PREPROCESSED_FILEPATH += '/' + latest_pickle
    else:
        PREPROCESSED_FILEPATH = './cache/preprocessed/' + DF_VERSION + '.pickle'
    if not exists(PREPROCESSED_FILEPATH): raise Exception(f'Given preprocessed version {DF_VERSION} not exists.')
    with open(PREPROCESSED_FILEPATH, 'rb') as fp:
        train_df, test_df, full_sensor_list, df_status = pickle.load(fp)
