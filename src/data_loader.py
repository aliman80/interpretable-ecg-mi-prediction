import os
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ast

def download_ptbxl_csvs(data_dir='data/ptb-xl'):
    """
    Download only the metadata CSVs to avoid a 3GB full dataset download during testing.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    import urllib.request
    
    base_url = 'https://physionet.org/files/ptb-xl/1.0.3/'
    files = ['ptbxl_database.csv', 'scp_statements.csv']
    
    for f in files:
        file_path = os.path.join(data_dir, f)
        if not os.path.exists(file_path):
            print(f"Downloading {f}...")
            urllib.request.urlretrieve(base_url + f, file_path)
    print("CSV metadata downloaded.")

def load_raw_data(df, sampling_rate, path):
    """
    Load raw ECG signal data directly from PhysioNet.
    """
    data = []
    print("Streaming actual ECG records from PhysioNet...")
    for f in tqdm(df.filename_lr):
        try:
            # Stream from PhysioNet directly instead of local disk
            record_dir = '/'.join(f.split('/')[:-1])
            record_name = f.split('/')[-1]
            pn_path = f'ptb-xl/1.0.3/{record_dir}'
            record = wfdb.rdsamp(record_name, pn_dir=pn_path)
            data.append(record)
        except Exception as e:
            print(f"Failed to stream {f}: {e}")
            # Append zero array if stream fails to keep shape consistent
            data.append((np.zeros((1000, 12)), None))
            
    data = np.array([signal for signal, meta in data])
    return data

def prepare_data(data_dir='data/ptb-xl', sampling_rate=100, max_samples=None):
    """
    Load and preprocess the PTB-XL dataset for MI classification.
    """
    # Load dataset metadata
    df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    if max_samples is not None:
        # Ensure all folds have some data
        samples_per_fold = max(1, max_samples // 10)
        df = df.groupby('strat_fold').head(samples_per_fold)
        print(f"Subsampled to {len(df)} records across all folds.")
    
    # Load raw signal data
    print(f"Loading raw ECG data (sampling rate: {sampling_rate} Hz)...")
    X = load_raw_data(df, sampling_rate, data_dir)
    
    # Load SCP statements to map diagnostic labels
    agg_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    # Define a function to aggregate diagnostic classes
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    # Apply mapping
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    
    # Create binary MI label: 1 if 'MI' is in the superclass, else 0
    df['label'] = df.diagnostic_superclass.apply(lambda x: 1 if 'MI' in x else 0)
    
    # Use the recommended splits from PTB-XL
    # folds 1-8: train, fold 9: val, fold 10: test
    train_fold = [1, 2, 3, 4, 5, 6, 7, 8]
    val_fold = [9]
    test_fold = [10]
    
    X_train = X[df.strat_fold.isin(train_fold)]
    y_train = df[df.strat_fold.isin(train_fold)]['label'].values
    
    X_val = X[df.strat_fold.isin(val_fold)]
    y_val = df[df.strat_fold.isin(val_fold)]['label'].values
    
    X_test = X[df.strat_fold.isin(test_fold)]
    y_test = df[df.strat_fold.isin(test_fold)]['label'].values
    
    df_test = df[df.strat_fold.isin(test_fold)].copy()
    
    # Normalize data (Standardization per lead across the training set)
    # X shape is (samples, sequence_length, leads)
    print("Normalizing data...")
    mean = np.mean(X_train, axis=(0, 1))
    std = np.std(X_train, axis=(0, 1))
    
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # PyTorch expects channels first: (samples, leads, sequence_length)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    
    return X_train, y_train, X_val, y_val, X_test, y_test, df_test

if __name__ == "__main__":
    download_ptbxl_csvs()
