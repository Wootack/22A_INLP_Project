import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

import macros


def isBinarySensor(sensor):
    return sensor not in [macros.T001, macros.T002, macros.T003, macros.T004, macros.T005, macros.ENTERHOME, macros.LEAVEHOME]
    # return sensor not in ['T001', 'T002', 'T003', 'T004', 'T005']


def isTemperatureSensor(sensor):
    return sensor in [macros.T001, macros.T002, macros.T003, macros.T004, macros.T005, macros.ENTERHOME, macros.LEAVEHOME]
    # return sensor in ['T001', 'T002', 'T003', 'T004', 'T005']


def remove_temperature_from_df(df):
    new_df = df[df['sensor'].apply(isBinarySensor)]
    return new_df


def temp_save(content):
    with open('temp.pickle', 'wb') as fp:
        pickle.dump(content, fp, pickle.HIGHEST_PROTOCOL)


def generate_scaler(scaler_type):
    if scaler_type == 'Standard': return StandardScaler()
    elif scaler_type == 'MinMax': return MinMaxScaler()
    elif scaler_type == 'MaxAbs': return MaxAbsScaler()
    elif scaler_type == 'Robust': return RobustScaler()
    else: raise Exception(f'Invalid Scaler Type {scaler_type} Given.')