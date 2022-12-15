import shutil
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from datetime import timedelta
from time import time, ctime
from traceback import print_exc
import logging

import pandas as pd

import macros
from testbench import TestBench
from detector import Detector
import configurations
from trim_data import parse_data_to_csv, preprocess_data

###################################################
# Paths
ARUBA_SMARTHOME_DATA_PATH = Path('.') / 'aruba_smart_home_dataset'
RAW_DATA_FILEPATH = ARUBA_SMARTHOME_DATA_PATH / 'data'
PROCESSED_DATA_FILEPATH = ARUBA_SMARTHOME_DATA_PATH / 'processed.csv'
DIRPATH_RESULTS = Path('test_results')
DIRPATH_CACHE = Path('.') / 'cache'
DIRPATH_PREPROCESSED = DIRPATH_CACHE / 'preprocessed'
DF_VERS_FILEPATH = DIRPATH_PREPROCESSED / 'df_versions.csv'
DETECTOR_FILEPATH = DIRPATH_CACHE / 'detector.pickle'
DIRPATH_SAVED_MODELS = Path('.') / 'saved_models'

DEBUG_FILEPATH = Path('.') / 'debug'
os.makedirs(DEBUG_FILEPATH, exist_ok=True)

os.makedirs(DIRPATH_SAVED_MODELS, exist_ok=True)
os.makedirs(DIRPATH_CACHE, exist_ok=True)
os.makedirs(DIRPATH_RESULTS, exist_ok=True)
os.makedirs(DIRPATH_RESULTS/'00000', exist_ok=True)

def get_curr_execution_num():
    global DIRPATH_RESULTS
    dirs = os.listdir(DIRPATH_RESULTS)
    dirs = sorted([int(dir) for dir in dirs if os.path.isdir(os.path.join(DIRPATH_RESULTS, dir))])
    curr = dirs[-1] + 1 if dirs else 0
    curr = str(curr).zfill(5)
    DIRPATH_RESULTS = DIRPATH_RESULTS / curr
    os.makedirs(DIRPATH_RESULTS)
    shutil.copyfile("./configurations.py", str(DIRPATH_RESULTS)+"/configurations.txt")

def preprocess(args):
    print("Parsing into CSV...")
    df = parse_data_to_csv(RAW_DATA_FILEPATH, PROCESSED_DATA_FILEPATH)
    print("Preprocessing data...")
    train_df, test_df, full_sensor_list, df_status = preprocess_data(df)
    os.makedirs(DIRPATH_PREPROCESSED, exist_ok=True)
    if os.path.exists(DF_VERS_FILEPATH):
        dfver_history = pd.read_csv(DF_VERS_FILEPATH, index_col=0, dtype=str, keep_default_na=False)
        curr_ver = 'df'+str(int(dfver_history.index[-1][2:])+1).zfill(3)
        for idx, row in dfver_history.iterrows():
            no_match = False
            for key, val in df_status.items():
                if row[key]!=val:
                    no_match = True
                    break
            if no_match: continue
            curr_ver = idx
            break
        if no_match:
            pd.concat([dfver_history, pd.DataFrame(df_status, index=[curr_ver])], axis=0, ignore_index=False).to_csv(DF_VERS_FILEPATH)
    else:
        curr_ver = 'df000'
        pd.DataFrame(df_status, index=['df000']).to_csv(DF_VERS_FILEPATH)
    df_status['df_version'] = curr_ver
    with open(DIRPATH_PREPROCESSED/(curr_ver+'.pickle'), 'wb') as fp:
        pickle.dump((train_df, test_df, full_sensor_list, df_status), fp, pickle.HIGHEST_PROTOCOL)


def debugg(args):
    TestBench(True, {macros.M009:['Missing'], macros.M020:['Missing']})
    print('Debugging')


def train_detector(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(DIRPATH_RESULTS/'detector.log'))
    logger.debug('Generating Detector')
    detector = Detector(DIRPATH_RESULTS, logger)
    detector.train()
    detector.test()
    # with DETECTOR_FILEPATH.open('wb') as fp:
        # pickle.dump(detector, fp, pickle.HIGHEST_PROTOCOL)


def parse_arguments():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_preprocess = subparsers.add_parser('preprocess')
    parser_preprocess.set_defaults(func=preprocess)

    parser_debugg = subparsers.add_parser('debugg')
    parser_debugg.set_defaults(func=debugg)

    parser_detector = subparsers.add_parser('detector')
    parser_detector.set_defaults(func=train_detector)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    err = False
    start = time()
    get_curr_execution_num()
    with open(DIRPATH_RESULTS/'README.txt', 'w') as f:
        f.write("Start time : ")
        f.write(ctime(start))
    try:
        args = parse_arguments()
        args.func(args)
    except Exception as e:
        print_exc()
        err = True
    end = time()
    with open(DIRPATH_RESULTS/'README.txt', 'a') as f:
        f.write("\nTerminated : ")
        f.write(ctime(end))
        f.write("\nTime spent : ")
        f.write(str(timedelta(seconds=(end-start))))
        f.write("\n")
        if err : shutil.rmtree(DIRPATH_RESULTS)