from pathlib import Path
import os
import re
import numpy as np
import datetime

from typing import Dict, List
import pandas as pd
from tqdm.auto import tqdm

import macros
from functions import isTemperatureSensor, remove_temperature_from_df, isBinarySensor
from configurations import (    RMV_NUMERIC_SENSORS, RMV_OUTLIERS, CRITERION_FOR_OUTLIERS, RMV_NON_UNIQUE_TIMESTAMP, NON_UNIQUE_TIMESTAMP_INDEX_RANGE,
                                HANDLE_SUCCESSIVE_BINARY_VALUE, TARGET_MOTIONS, TRAIN_DATA_BORDER, TEST_DATA_BORDER)


def parse_data_to_csv(RAW_DATA_FILEPATH: Path, PROCESSED_DATA_FILEPATH: Path) -> pd.DataFrame:
    if not os.path.exists(PROCESSED_DATA_FILEPATH):
        line_pattern = re.compile(
            r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\.\d{0,6})?)\s+(\w+)\s+(.+)$')
        datetime_pattern = re.compile(
            r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\.?(\d{0,6})$')
        datetime_format = '%Y-%m-%d %H:%M:%S.%f'
        event_pattern = re.compile(r'^([A-Za-z0-9.]+)(?:\s+(\S.*))?$')

        def pad_microseconds(match):
            microsecond_str = match.group(2)
            if microsecond_str is None:
                microsecond_str = ''
            microsecond_str = microsecond_str.ljust(6, '0')
            return match.group(1) + '.' + microsecond_str

        with RAW_DATA_FILEPATH.open() as fp:
            lines = list(fp)

        rows = []
        for line in tqdm(lines):
            line_pattern_match = line_pattern.match(line.strip())
            datetime_str = line_pattern_match.group(1)
            datetime_str = datetime_pattern.sub(pad_microseconds, datetime_str)
            timestamp = datetime.datetime.strptime(datetime_str, datetime_format)
            sensor = line_pattern_match.group(2)
            event_str = line_pattern_match.group(3)
            event_match = event_pattern.match(event_str)
            event = event_match.group(1)
            label = event_match.group(2)
            rows.append({'timestamp': timestamp, 'sensor': sensor,
                        'event': event, 'label': label})

        df = pd.DataFrame(rows)
        df = df[df['sensor'] != 'c']
        df.loc[(df['event'] == 'O') & (
            df['label'] == 'Relax begin'), 'event'] = 'ON'
        df.loc[(df['label'] == 'Relax  begin'), 'label'] = 'Relax begin'
        df.loc[(df['label'] == 'Meal_Preparation  begin'),
               'label'] = 'Meal_Preparation begin'
        df.loc[(df['label'] == 'Meal_Preparation  end'),
               'label'] = 'Meal_Preparation end'
        df.loc[:, 'event'] = (df['event'].replace(
            [r'^O.*F.*$', r'^ON.*$', r'^OPENc$', 'CLOSED',
                r'^([0-9]{2})cc$', r'^([0-9]{2})5$', r'^([0-9]{2}\.[0-9]).+$'],
            ['OFF', 'ON', 'OPEN', 'CLOSE', r'\1', r'\1.5', r'\1'],
            regex=True))
        df.loc[:, 'label'] = df['label'].replace(r'[\.5c]+', '', regex=True)
        df = df.iloc[:, :3]
        df.to_csv(PROCESSED_DATA_FILEPATH, index=False) # store preprocessed csv
    df = pd.read_csv(PROCESSED_DATA_FILEPATH, parse_dates=['timestamp'], dayfirst=False, infer_datetime_format=True)
    return df



def preprocess_data(df):
    '''
        Returns
        -------
        full dataframe : full data for target motions
    '''
    if RMV_NON_UNIQUE_TIMESTAMP:
        df = df.drop(index=range(NON_UNIQUE_TIMESTAMP_INDEX_RANGE[0], NON_UNIQUE_TIMESTAMP_INDEX_RANGE[1]))
        df.reset_index(drop=True, inplace=True)
    df['event'].replace('ON', True, inplace=True)
    df['event'].replace('OPEN', True, inplace=True)
    df['event'].replace('OFF', False, inplace=True)
    df['event'].replace('CLOSE', False, inplace=True)
    df['sensor'] = df['sensor'].map(macros.sensor_dict)
    if RMV_NUMERIC_SENSORS:
        df = remove_temperature_from_df(df)
        df.reset_index(drop=True, inplace=True)
    df = drop_successive_events(df)
    full_sensor_list = df.sensor.sort_values().unique()
    train_start, train_end = TRAIN_DATA_BORDER
    train_start = datetime.datetime(*train_start)
    train_end = datetime.datetime(*train_end)
    train_df = df[ (df['timestamp'] > train_start) & (df['timestamp'] < train_end) ]
    train_df.reset_index(drop=True, inplace=True)
    test_start, test_end = TEST_DATA_BORDER
    test_start = datetime.datetime(*test_start)
    test_end = datetime.datetime(*test_end)
    test_df = df[ (df['timestamp'] > test_start) & (df['timestamp'] < test_end) ]
    test_df.reset_index(drop=True, inplace=True)
    df_status = {   'Remove Numeric'                    :   str(RMV_NUMERIC_SENSORS),
                    'Remove Outliers'                   :   str(RMV_OUTLIERS),
                    'Criterion for outliers'            :   str(CRITERION_FOR_OUTLIERS),
                    'Remove non-unique timestamp'       :   str(RMV_NON_UNIQUE_TIMESTAMP),
                    'Non-unique timestamp index range'  :   str(NON_UNIQUE_TIMESTAMP_INDEX_RANGE),
                    'Handling successive binary values' :   str(HANDLE_SUCCESSIVE_BINARY_VALUE),
                    'Target motions'                    :   str(TARGET_MOTIONS),
                    'Room-sensor clustering'            :   str(ROOM_SENSORS),
                    'Train and test data border'        :   str((TRAIN_DATA_BORDER, TEST_DATA_BORDER))  }
    return train_df, test_df, full_sensor_list, df_status


def drop_successive_events(df):
    full_sensor_list = sorted(df.sensor.unique())
    if HANDLE_SUCCESSIVE_BINARY_VALUE == 'Leave First':
        drop_i = []
        for sensor in full_sensor_list:
            if not isBinarySensor(sensor): continue
            sen_df = df[df['sensor']==sensor]['event']
            if sen_df.shape[0]<2: continue
            curr = not sen_df.iloc[0]
            for i, value in sen_df.iteritems():
                if curr==value:
                    drop_i.append(i)
                curr = value
        df.drop(drop_i, inplace=True)
        df.sort_values(by='timestamp', ignore_index=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        raise Exception('Unimplemented Error : Handling succesive binary value')