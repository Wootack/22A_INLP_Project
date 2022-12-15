import random
from datetime import timedelta

import pandas as pd
import numpy as np

from configurations import RANDOM_STATE
from common_vars import full_sensor_list


UNIFORM_MISSING             = 0
MISSING                     = 1
ON_FLICKERING               = 2
UNIFORM_FLICKERING_SECOND   = 3


def inject_uniform_missing(df, error_sensor, missing_ratio):
    drop_i = []
    error_timestamps = []
    sen_df = df[(df['sensor']==error_sensor)]['event']
    drop_false = False
    random.seed(RANDOM_STATE + list(full_sensor_list).index(error_sensor) + UNIFORM_MISSING + int(missing_ratio * 100))
    for i, value in sen_df.iteritems():
        if value:
            if random.random() < missing_ratio:
                drop_i.append(i)
                error_timestamps.append(df.at[i, 'timestamp'])
                drop_false = True
        if not value and drop_false:
            drop_i.append(i)
            error_timestamps.append(df.at[i, 'timestamp'])
            drop_false = False
    new_df = df.drop(drop_i)
    new_df.reset_index(drop=True, inplace=True)
    return new_df, error_timestamps

def inject_missing(df, error_sensor):
    new_df = df[df['sensor']!=error_sensor]
    new_df.reset_index(drop=True, inplace=True)
    error_timestamps = df[df['sensor']==error_sensor]['timestamp'].tolist()
    return new_df, error_timestamps

def inject_on_flickering(df, error_sensor, flickering_ratio, flickering_max):
    random.seed(RANDOM_STATE + list(full_sensor_list).index(error_sensor) + ON_FLICKERING + int((flickering_max+flickering_ratio) * 100))
    normal_df = df[(df['sensor'] != error_sensor)]
    error_df = df[(df['sensor'] == error_sensor)].reset_index(drop=True)
    error_timestamps = []
    new_rows = []
    for idx, row in error_df.iterrows():
        if row['event']:
            if random.random() < flickering_ratio:
                if idx >= len(error_df)-1: break
                next_timestamp = error_df.at[idx+1, 'timestamp']
                label = row['label']
                cur_event = False
                num_flicker = 2*random.randrange(1, flickering_max)
                times = sorted([randomDate(row['timestamp'], next_timestamp) for _ in range(num_flicker)])
                for i in range(num_flicker):
                    new_rows.append([times[i], error_sensor, cur_event, label])
                    cur_event = not cur_event
                    error_timestamps.append(times[i])
    new_df = pd.DataFrame(new_rows, columns=['timestamp', 'sensor', 'event', 'label'])
    newDataFrame = pd.concat([normal_df, error_df, new_df]).sort_values(by='timestamp', ignore_index=True)
    return newDataFrame, error_timestamps


def inject_uniform_flickering_second(df, error_sensor, period):
    random.seed(RANDOM_STATE + list(full_sensor_list).index(error_sensor) + UNIFORM_FLICKERING_SECOND + int(period * 100))
    np.random.seed(RANDOM_STATE + list(full_sensor_list).index(error_sensor) + UNIFORM_FLICKERING_SECOND + int(period * 100))
    normal_df = df[df['sensor'] != error_sensor]
    error_df = df[df['sensor'] == error_sensor].reset_index(drop=True)
    new_rows = []
    error_timestamps = []
    for idx, row in error_df.iterrows():
        if row['event']:
            if idx >= len(error_df)-1: break
            next_timestamp = error_df.at[idx+1, 'timestamp']
            duration = next_timestamp - row['timestamp']
            gen_num = int( (duration / period).total_seconds() ) * 2
            if gen_num == 0: continue
            gen_time = sorted([row['timestamp'] + timedelta(seconds=(i+1)*period/2 + 0.1*period*np.random.randn()) for i in range(gen_num)])
            while True:
                if gen_time[-1] > next_timestamp:
                    gen_time[-1] = randomDate(row['timestamp'] + timedelta(seconds=gen_num*period/2), next_timestamp)
                    gen_time.sort()
                else: break
            cur_event = False
            for i in range(gen_num):
                new_rows.append([gen_time[i], error_sensor, cur_event, row['label']])
                cur_event = not cur_event
                error_timestamps.append(gen_time[i])
    new_df = pd.DataFrame(new_rows, columns=['timestamp', 'sensor', 'event', 'label'])
    new_df = pd.concat([normal_df, error_df, new_df]).sort_values(by='timestamp', ignore_index=True)
    return new_df, error_timestamps


def injectError(df, error_sensor_typelist_dict):
    error_sensor_timestamps = {}
    for error_sensor, error_typelist in error_sensor_typelist_dict.items():
        error_sensor_timestamps[error_sensor] = []
        for error_type in error_typelist:
            splitted = error_type.split('/')
            if splitted[0] == 'Uniform_Missing':
                df, error_timestamps = inject_uniform_missing(df, error_sensor, missing_ratio=float(splitted[1]))
            elif splitted[0] == 'Missing':
                df, error_timestamps = inject_missing(df, error_sensor)
            elif splitted[0] == 'On_Flickering':
                df, error_timestamps = inject_on_flickering(df, error_sensor, flickering_ratio=float(splitted[1]), flickering_max=int(splitted[2]))
            elif splitted[0] == 'Uniform_Flickering_Second':
                df, error_timestamps = inject_uniform_flickering_second(df, error_sensor, period=float(splitted[1]))
            else: raise Exception(f'Invalid error type {error_type} given')
            error_sensor_timestamps[error_sensor] += error_timestamps
        error_sensor_timestamps[error_sensor].sort()
    return df, error_sensor_timestamps
    

def randomDate(start_time, end_time):
    deltatime = end_time-start_time
    return start_time + random.random() * deltatime

def randomSensorValue(sensor):
    if sensor in ['T001', 'T002', 'T003', 'T004', 'T005']:
        return random.randint(0, 100)
    else:
        return bool(random.getrandbits(1))

