from itertools import combinations, chain
from functools import partial
from bisect import bisect_left

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import macros
from configurations import  TRAINED_ERR_TYPES, TEST_ERR_TYPES, TRAIN_SIMUL_ERR_SENSORS,\
        TEST_SIMUL_ERR_SENSORS, TRAIN_ERROR_THRESHOLD, TEST_ERROR_THRESHOLD, TEST_SAMPLE_SIZE,\
        TARGET_SENSOR_LIST, TRAIN_SAMPLE_SIZE, RANDOM_STATE, SEQ_LEN,\
        BATCH_SIZE, HIDDEN_SIZE, NUM_LAYER, LEARNING_RATE, EPOCH_NUM
from common_vars import df_status, full_sensor_list
from testbench import TestBench
from functions import temp_save, generate_scaler

class Detector:
    def __init__(self, DIRPATH_TEST_RESULTS, logger):
        self.target_sensor_list = TARGET_SENSOR_LIST
        self.DIRPATH_TEST_RESULTS = DIRPATH_TEST_RESULTS
        self.logger = logger
        self.trainer = None
        self.threshold = None
        self.scaler = None
        logger.info('{:<46} : {}'.format('Detector with preprocessed data version', df_status["df_version"]))
        logger.info('{:<46} : {}'.format('Error type used for training', TRAINED_ERR_TYPES))
        logger.info('{:<46} : {}'.format('Error type used for testing', TEST_ERR_TYPES))
        logger.info('{:<46} : {}'.format('Simultaneous sensor error for training', str(TRAIN_SIMUL_ERR_SENSORS[0]) + ' to ' + str(TRAIN_SIMUL_ERR_SENSORS[1])))
        logger.info('{:<46} : {}'.format('Simultaneous sensor error for testing', str(TEST_SIMUL_ERR_SENSORS[0]) + ' to ' + str(TEST_SIMUL_ERR_SENSORS[1])))
        logger.info('{:<46} : {}'.format('Threshold for train error data', TRAIN_ERROR_THRESHOLD))
        logger.info('{:<46} : {}'.format('Threshold for test error data', TEST_ERROR_THRESHOLD))
        logger.info('{:<46} : {}'.format('Training data sampling size', TRAIN_SAMPLE_SIZE))
        logger.info('{:<46} : {}'.format('Testing data sampling size', TEST_SAMPLE_SIZE))
        logger.info('{:<46} : {}'.format('Threshold for test error data', TRAIN_ERROR_THRESHOLD))
        logger.info('{:<46} : {}'.format('LSTM batch size', BATCH_SIZE))
        logger.info('{:<46} : {}'.format('LSTM hidden layer size', HIDDEN_SIZE))
        logger.info('{:<46} : {}'.format('LSTM number of layers', NUM_LAYER))
        logger.info('{:<46} : {}'.format('LSTM learning rate', LEARNING_RATE))
        logger.info('{:<46} : {}'.format('LSTM number of epochs', EPOCH_NUM))
        logger.info('{:=<100}\n{:=<100}\n'.format('',''))

    
    def train(self):
        self.logger.info('Training Start ...')
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            self.logger.warning('{:!^100}\n'.format(' GPU not available. Running on CPU '))
            device = torch.device("cpu")
        self.device = device
        train_loader = self.collect_input_dataset(train=True)
        model = LSTM1(  num_classes=len(TARGET_SENSOR_LIST),
                        input_size=len(full_sensor_list)+1,
                        hidden_size=HIDDEN_SIZE,
                        num_layers=NUM_LAYER,
                        seq_length=SEQ_LEN)
        model.to(device)
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_history = []
        for epoch in trange(EPOCH_NUM):
            ep_loss = 0
            model.train()
            for idx, data in enumerate(train_loader):
                X, y = data
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                preds = model(X)
                loss = criterion(preds, y)
                loss.backward()
                ep_loss += loss
                optimizer.step()
            ep_loss /= (len(train_loader) * BATCH_SIZE)
            loss_history.append(ep_loss.item())
        self.model = model
        plot_loss_epoch(self.DIRPATH_TEST_RESULTS, loss_history)


    def test(self):
        self.logger.info('\nTesting Start ...')
        test_loader = self.collect_input_dataset(train=False)
        model = self.model
        device = self.device
        model.to(device)
        model.eval()
        label_score = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                X, y = data
                X = X.to(device)
                y = y.to(device)
                preds = model(X)
                label_score += zip(y.cpu().data.numpy(),
                                    preds.cpu().data.numpy())
        temp_save(label_score)
        labels, preds = zip(*label_score)
        labels = np.array(labels)
        scores = torch.sigmoid(torch.tensor(preds)).data.numpy()
        self.logger.info('\n{:=^100}'.format(' Results '))
        for i, sen in enumerate(TARGET_SENSOR_LIST):
            sensor = macros.sensor_dict.inverse[sen]
            auc = roc_auc_score(labels[:,i], scores[:,i])
            self.logger.info('ROC_AUC Score of {:<6} Sensor : {:<10}'.format(sensor, auc))


    def collect_input_dataset(self, train):
        np.random.RandomState(RANDOM_STATE)
        tb = TestBench(train, {})
        sample_size = TRAIN_SAMPLE_SIZE if train else TEST_SAMPLE_SIZE
        idxs = np.random.choice(tb.features.shape[0]-SEQ_LEN, sample_size, replace=False)
        X = []
        for idx in idxs:
            X.append(np.array(tb.features.iloc[idx:idx+SEQ_LEN, :]))
        y = [[0, 0, 0]] * len(X)
        prev_error = len(X)
        self.logger.info('\n{:=^100}'.format(' Number of Data '))
        self.logger.info('{:<36} Data Count :{}'.format('Normal', prev_error))

        simul_err_sensors = TRAIN_SIMUL_ERR_SENSORS if train else TEST_SIMUL_ERR_SENSORS
        error_types = TRAINED_ERR_TYPES if train else TEST_ERR_TYPES
        partial_comb = partial(combinations, TARGET_SENSOR_LIST)
        error_cases = list(chain(*map(partial_comb, range(simul_err_sensors[0], simul_err_sensors[1]+1))))
        for err_sen_tup in error_cases:
            err_dict = {err_sen:error_types for err_sen in err_sen_tup}
            tb = TestBench(train, err_dict)
            features = tb.features
            err_cnts = tb.error_counts
            idxs = get_idxs_with_enough_errors(features, err_cnts, err_sen_tup, sample_size)
            prev_error = len(X)
            cur_label = [1 if sen in err_sen_tup else 0 for sen in TARGET_SENSOR_LIST]
            for idx in idxs:
                X.append(np.array(features.iloc[idx:idx+SEQ_LEN, :]))
                y.append(cur_label)
            self.logger.info('{:<36} Data Count :{}'.format('Errors on '+\
                    str([macros.sensor_dict.inverse[sen] for sen in err_sen_tup]), len(X) - prev_error))
        self.logger.info('{:<36} Data Count :{}'.format('Total', len(X)))
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        dataset = sensor_data(X, y)
        data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        return data_loader


def get_idxs_with_enough_errors(features, err_cnts, err_sen_tup, sample_size):
    cand1 = err_cnts
    for err_sen in err_sen_tup:
        cand1 = cand1[cand1[err_sen]!=0]
    filtered = cand1
    for err_sen in err_sen_tup:
        q3 = np.quantile(filtered[err_sen], TRAIN_ERROR_THRESHOLD)
        filtered = filtered[filtered[err_sen]>q3]
    val_data_num = filtered.shape[0]
    if val_data_num > sample_size:
        idxs = np.random.choice(val_data_num, sample_size, replace=False)
    else: idxs = np.arange(val_data_num)
    idxs = filtered.index[idxs]
    idxs = [bisect_left(features.index, idx) for idx in idxs]
    return idxs


def plot_loss_epoch(save_path, loss_history):
        plt.figure()
        plt.plot(range(EPOCH_NUM), loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Average Training Loss')
        plt.title('AVG Training Loss - Epochs Graph')
        plt.savefig(save_path/'AVG_Loss.png')
        plt.figure()
        plt.plot(range(EPOCH_NUM//2, EPOCH_NUM), loss_history[EPOCH_NUM//2:])
        plt.xlabel('Epochs')
        plt.ylabel('Average Training Loss')
        plt.title('AVG Training Loss - Epochs Graph for Rear Half')
        plt.savefig(save_path/'AVG_Loss_Rear_Half.png')


class sensor_data(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.len = x.shape[0]


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


    def __len__(self):
        return self.len


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        output, _status =self.lstm(x)
        output = output[:,-1,:]
        out = self.relu(output)
        out = self.fc(out)
        return out