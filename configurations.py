import macros
from easydict import EasyDict
from sys import argv

''' =========================================================================================================================================
Preprocess
'''
# Remove numeric sensor data. In this case, Temperature sensors are removed.
RMV_NUMERIC_SENSORS = True

# Remove outliers.
RMV_OUTLIERS = False

# Criterion for outliers.
CRITERION_FOR_OUTLIERS = ''

# Remove data with non-unique timestamp.
RMV_NON_UNIQUE_TIMESTAMP = True

# Specify the range of index of non-unique timestamp.
NON_UNIQUE_TIMESTAMP_INDEX_RANGE = (1564861, 1571286)

# Handle successive true or false.
HANDLE_SUCCESSIVE_BINARY_VALUE = 'Leave First'
if HANDLE_SUCCESSIVE_BINARY_VALUE not in ['Leave First', 'Leave Last','Add Random Timestamp From Front', 'Add Random Timestamp From Last', 'Leave All']:
    raise Exception(f'Invalid value for HANDLE_SUCCESSIVE_BINARY_VALUE : {HANDLE_SUCCESSIVE_BINARY_VALUE}')
if HANDLE_SUCCESSIVE_BINARY_VALUE != 'Leave First': raise Exception(f'Unimplemented error : handling {HANDLE_SUCCESSIVE_BINARY_VALUE}')

# Motions not involved in this list will be removed.
TARGET_MOTIONS = ['Relax', 'Meal_Preparation', 'Sleeping', 'Eating', 'Work', 'Housekeeping', 'Wash_Dishes']

# Borders of training and test dataset
TRAIN_DATA_BORDER = [(2010, 11, 4, 0, 3, 50, 209588), (2011, 5, 1, 2, 47, 32, 40924)]
TEST_DATA_BORDER = [(2011, 5, 1, 2, 47, 32, 40925), (2011, 6, 11, 23, 58, 10, 4835)]

''' =========================================================================================================================================
TestBench
'''
# Determine the version of preprocessed dataframe.
DF_VERSION = 'df000'

# Error detector will only consider sensors involved in this list.
TARGET_SENSOR_LIST = [macros.M009, macros.M013, macros.M020]

''' =========================================================================================================================================
Detector
'''
# Select model used for detector.
DETECTOR_TYPE = 'LSTM'
if DETECTOR_TYPE not in ['LSTM']:
    raise Exception(f'Invalid value for DETECTOR_TYPE : {DETECTOR_TYPE}')

TRAIN_SAMPLE_SIZE = 10000
TEST_SAMPLE_SIZE = 10000

SEQ_LEN = 100
BATCH_SIZE = 128
HIDDEN_SIZE = 20
NUM_LAYER = 3
LEARNING_RATE = 0.001
EPOCH_NUM = 500


RANDOM_STATE = 42


# Detector will train with these error types injected.
# 'Missing'
# 'Uniform_Missing/MISSING_RATIO'
# 'On_Flickering/FLICKERING_RATIO/MAX_FLICKER_NUM'
# 'Uniform_Flickering_Second/PERIOD'
# 'Single_All'
TRAINED_ERR_TYPES = ['Uniform_Missing/0.5']
TEST_ERR_TYPES = ['Uniform_Missing/0.5']

TRAIN_SIMUL_ERR_SENSORS = (3, 3)
TEST_SIMUL_ERR_SENSORS = (3, 3)

# Determine how much change of data will be considered as proper error. Work only for CUT_BY_ADL is true.
TRAIN_ERROR_THRESHOLD = 0.2
TEST_ERROR_THRESHOLD = 0.2