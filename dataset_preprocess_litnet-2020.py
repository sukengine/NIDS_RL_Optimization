#LITNET-2020 Dataset Proprocess

import os
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.preprocessing import MinMaxScaler
import ipaddress

# define attack type
attack_types = {
    1: ("icmp_smf", "unused/SMURF_v2.csv"),
    2: ("icmp_f", "ICMP_FLOOD_v2.csv"),
    3: ("udp_f", "UDP_FLOOD_v2.csv"),
    4: ("tcp_syn_f", "SYN_FLOOD.csv"),
    5: ("http_f", "HTTP_FLOOD_v2.csv"),
    6: ("tcp_land", "LAND_ATTACK_v2.csv"),
    7: ("tcp_w32_w", "BLASTER_WORM_v2.csv"),
    8: ("tcp_red_w", "RED_WORM_v2.csv"),
    9: ("smtp_b", "SPAM_v2.csv"),
    10: ("udp_reaper_w", "REAPER_WORM_v2.csv"),
    11: ("tcp_udp_win_p", "SCANNING_SPREAD_v2.csv"),
    12: ("udp_0", "FRAGMENTATION_v2.csv")
}

# define csv header
header = [
    'ID', 'file', 'ts_year', 'ts_month', 'ts_day', 'ts_hour', 'ts_min', 'ts_second', 'te_year', 'te_month', 'te_day',
    'te_hour', 'te_min', 'te_second', 'td', 'sa', 'da', 'sp', 'dp', 'pr', '_flag1', '_flag2', '_flag3', '_flag4',
    '_flag5', '_flag6', 'fwd', 'stos', 'ipkt', 'ibyt', 'opkt', 'obyt', '_in', 'out', 'sas', 'das', 'smk', 'dmk',
    'dtos', '_dir', 'nh', 'nhb', 'svln', 'dvln', 'ismc', 'odmc', 'idmc', 'osmc', 'mpls1', 'mpls2', 'mpls3', 'mpls4',
    'mpls5', 'mpls6', 'mpls7', 'mpls8', 'mpls9', 'mpls10', 'cl', 'sl', 'al', 'ra', 'eng', 'exid', 'tr', 'attack_t'
]


# value mapping between value in the last column(attack_t) and attack class
value_mapping = ["none", "icmp_smf", "icmp_f", "udp_f", "tcp_syn_f", "http_f", "tcp_land", "tcp_w32_w", "tcp_red_w", "smtp_b", "udp_reaper_w", "tcp_udp_win_p", "udp_0"]

#columns to be removed, which contained 1 unique value
columns_to_remove = {"fwd", "mpls10", "mpls5", "mpls6", "mpls7", "mpls8", "mpls9", "ra", "cl", "sl", "al", 
                     "mpls3", "eng", "tr", "mpls4", "osmc", "mpls2", "mpls1", "opkt", "idmc", "odmc", "ismc", 
                     "dvln", "svln", "nhb", "nh", "_dir", "dmk", "smk", "obyt", "dtos"}

# path to original .csv files
input_folder = "../../electronics9050800/dataset/csv_original/"
intermediate_folder = "../dataset_preprocessed4/"  # Output folder before train_test_split
os.makedirs(intermediate_folder, exist_ok=True)

# convert 16 bit integer port number using One-Hot Encoding as 16-bit
def binary_one_hot(value, num_bits=16):
    binary_str = format(value, f'0{num_bits}b')  # Convert to binary string 16 bit
    return [int(bit) for bit in binary_str]      # convert to list of integer 0 and 1

# read every .csv in folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)

        # read .csv
        df = pd.read_csv(file_path, header=None)

        # find the mapping number to the file name on attack_types
        attack_label = None
        for key, value in attack_types.items():
            if value[1] == file_name:
                attack_label = key
                break

        # add the previous mapping number to the file name as column 2
        if attack_label is not None:
            df.insert(1, 'attack_type', attack_label)

        # drop unused columns
        df = df.drop(df.columns[-21:-2], axis=1)
        if df.shape[1] > 2:
            df = df.drop(df.columns[-1], axis=1)

        df.columns = header

        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

        # create new field, duration
        df['start_time'] = pd.to_numeric(pd.to_datetime(df[['ts_year', 'ts_month', 'ts_day', 'ts_hour', 'ts_min', 'ts_second']]
                                        .rename(columns={'ts_year': 'year', 'ts_month': 'month', 'ts_day': 'day', 
                                                        'ts_hour': 'hour', 'ts_min': 'minute', 'ts_second': 'second'})))
        df['end_time'] = pd.to_numeric(pd.to_datetime(df[['te_year', 'te_month', 'te_day', 'te_hour', 'te_min', 'te_second']]
                                        .rename(columns={'te_year': 'year', 'te_month': 'month', 'te_day': 'day', 
                                                        'te_hour': 'hour', 'te_min': 'minute', 'te_second': 'second'})))

        df.insert(2, 'duration', (df['end_time'] - df['start_time']).astype('int'))
        # df['duration'] = (df['end_time'] - df['start_time']).astype('int')

        df.drop(columns=['ts_year', 'ts_month', 'ts_day', 'ts_hour', 'ts_min', 'ts_second', 
                        'te_year', 'te_month', 'te_day', 'te_hour', 'te_min', 'te_second', 'start_time', 'end_time'], inplace=True)

        # Check and change value of the last column
        last_column = df.columns[-1]
        df[last_column] = df[last_column].apply(lambda x: value_mapping.index(x) if x in value_mapping else np.nan)


        def is_ipv6(value):
            try:
                ipaddress.IPv6Address(value)
                return True
            except ipaddress.AddressValueError:
                return False

        # check value in dataframe, convert to float if they're not already float
        def safe_convert_to_float(value):
            try:
                # Check if it's ip address
                if isinstance(value, str) and all(part.isdigit() for part in value.split('.') if '.' in value):
                    # ip address to hash
                    hash_object = hashlib.md5(value.encode())
                    return float(int(hash_object.hexdigest(), 16) % 10**8)  # reduce hash
                # if isinstance(value, str) and is_ipv6(value):
                #     hash_object = hashlib.md5(value.encode())
                #     return float(int(hash_object.hexdigest(), 16) % 10**8)  # reduce hash to float
                # Convert other non-float to hash
                if isinstance(value, str):
                    hash_object = hashlib.md5(value.encode())
                    return float(int(hash_object.hexdigest(), 16) % 10**8)  # reduce hash to float
                return float(value)
            except ValueError:
                return np.nan

        columns_to_skip = ['ID', 'file', 'attack_t', 'sp', 'dp']
        columns_to_transform = [col for col in df.columns if col not in columns_to_skip]

        df[columns_to_transform] = df[columns_to_transform].applymap(lambda x: safe_convert_to_float(x) if not isinstance(x, float) else x)

        #Normalize data
        #exclude columns before normalize
        exclude_columns = ['ID', 'file', 'attack_t', 'sp', 'dp']

        columns_to_normalize = [col for col in df.columns if col not in exclude_columns]

        scaler = MinMaxScaler()

        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


        # Binary One-hot encoder
        if 'sp' in df.columns and 'dp' in df.columns:
            df_one_hot = df.copy()

            # locate 'sp' and 'dp'
            sp_index = df.columns.get_loc("sp")
            dp_index = df.columns.get_loc("dp")

            # convert 'sp' and 'dp'
            df_one_hot[['sp_bin', 'dp_bin']] = df[['sp', 'dp']].applymap(lambda x: binary_one_hot(int(x)))

            # create each One-Hot
            sp_cols = [f"sp_{i}" for i in range(16)]
            dp_cols = [f"dp_{i}" for i in range(16)]

            sp_df = pd.DataFrame(df_one_hot['sp_bin'].tolist(), index=df.index, columns=sp_cols)
            dp_df = pd.DataFrame(df_one_hot['dp_bin'].tolist(), index=df.index, columns=dp_cols)

            # remove original 'sp' และ 'dp'
            df_one_hot.drop(columns=['sp_bin', 'dp_bin', 'sp', 'dp'], inplace=True)

            # insert newly create One-Hot at the same location
            for i, col in enumerate(sp_cols):
                df_one_hot.insert(sp_index + i, col, sp_df[col])

            for i, col in enumerate(dp_cols):
                df_one_hot.insert(dp_index + i + 15, col, dp_df[col])  # add 15 because 'sp' has 16 new columns


        # write the preprocessed dataset in intermediate_folder before train_test_split
        intermediate_file_path = os.path.join(intermediate_folder, file_name)
        df_one_hot.to_csv(intermediate_file_path, index=False)


print("Processing complete. Train and test datasets created.")
