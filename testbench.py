import os
import pickle
from pathlib import Path

import macros
from feature_extract import extract_features
from error_inject import injectError
from common_vars import train_df, test_df, df_status
from functions import temp_save

FOLDER_PATH = Path('.') / 'testbenches'

class TestBench:
    def __init__(self, train, error_sensor_typelist_dict):
        df = train_df if train else test_df
        errors_str = ''
        for error_sensor, error_typelist in error_sensor_typelist_dict.items():
            error_str = '(' + macros.sensor_dict.inverse[error_sensor] + error_types_to_str(error_typelist) + ')'
            errors_str += error_str
        errors_str = 'Normal' if len(errors_str)==0 else errors_str
        testbench_dir = FOLDER_PATH / 'testbenches' / df_status['df_version'] / errors_str
        os.makedirs(testbench_dir, exist_ok=True)
        pickle_name = 'train.pickle' if train else 'test.pickle'
        testbench_file = testbench_dir / pickle_name
        loaded = self.load_data(testbench_file)
        if not loaded:
            df, error_sensor_timestamps_dict = injectError(df, error_sensor_typelist_dict)
            self.features, self.error_counts = extract_features(df, error_sensor_timestamps_dict)
            self.save_data(testbench_file)


    def load_data(self, testbench_file):
        if os.path.exists(testbench_file):
            with testbench_file.open('rb') as fp:
                self.features, self.error_counts = pickle.load(fp)
            return True
        return False


    def save_data(self, testbench_file):
        with testbench_file.open('wb') as fp:
            pickle.dump((self.features, self.error_counts), fp, pickle.HIGHEST_PROTOCOL)
    

def error_types_to_str(error_types):
    err_type_str = ''
    for error_type in error_types:
        err_type_str += '-'
        splitted = error_type.split('/')
        if splitted[0] == 'Uniform_Missing':
            err_type_str += ('U' + splitted[1])
        elif splitted[0] == 'Missing':
            err_type_str += 'M'
        elif splitted[0] == 'On_Flickering':
            err_type_str += ('OF' + splitted[1] + 'max' + splitted[2])
        elif splitted[0] == 'Uniform_Flickering_Second':
            err_type_str += ('UFS' + splitted[1])
        else: raise Exception(f'Invalid Error type {splitted[0]} given.')
    return err_type_str


def generate_testbench(arg):
    return TestBench(*arg)