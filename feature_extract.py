import datetime

import numpy as np
import pandas as pd
from bisect import bisect_right

from common_vars import full_sensor_list
from configurations import SEQ_LEN


def extract_features(df, error_sensor_timestamps_dict):
    sen_num = len(full_sensor_list)
    dfs = []
    for sen in full_sensor_list:
        diff_df = df[df.sensor==sen].diff() # on~off and off~on time
        diff_df = diff_df[diff_df.event == -1] # on~off time only
        diff_ser = diff_df.timestamp.reset_index(drop=True)
        diff_flt = diff_ser.apply(datetime.timedelta.total_seconds)
        true_df = df[(df.sensor==sen) & (df.event)].copy().reset_index(drop=True)
        true_df['timediff'] = diff_flt
        true_df.set_index('timestamp', inplace=True)
        base_np = np.zeros((true_df.shape[0], sen_num+1))
        feature_df = pd.DataFrame(base_np,
                columns=[*full_sensor_list, 'timediff'],
                index=true_df.index
            )
        feature_df.loc[:, sen] = true_df.timediff
        dfs.append(feature_df)
    concated = pd.concat(dfs, axis=0).sort_index()
    concated = concated[~concated.isna().any(axis=1)]
    concated['timediff'] = concated.index.to_series().diff().iloc[1:].apply(datetime.timedelta.total_seconds)
    concated = concated.iloc[1:]
    idxs = concated.index
    base_np = np.zeros((concated.shape[0], len(error_sensor_timestamps_dict)))
    error_counts_df = pd.DataFrame(base_np,
            columns=error_sensor_timestamps_dict.keys(),
            index=idxs
        )
    for error_sensor, times in error_sensor_timestamps_dict.items():
        for timestamp in times:
           idx = bisect_right(idxs, timestamp)
           error_counts_df.iloc[idx].loc[error_sensor] += 1
    error_cum = error_counts_df.cumsum(axis=0)
    error_cum = error_cum.iloc[SEQ_LEN-1:].reset_index(drop=True) - \
        error_cum[:-100].shift(1,fill_value=0).reset_index(drop=True)
    error_cum = error_cum[:-1]
    error_cum.set_index(error_counts_df.index[:-SEQ_LEN], drop=True, inplace=True)
    return concated, error_cum
