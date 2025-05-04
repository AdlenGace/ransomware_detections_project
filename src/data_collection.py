# data_collection.py

import os
import pandas as pd

def collect_sysmon_logs(log_path):
    """
    Collect logs from Sysmon (or any other monitoring tool)
    Args:
        log_path (str): Path to the log file or directory
    Returns:
        pd.DataFrame: Collected logs in a pandas DataFrame
    """
    logs = []
    for filename in os.listdir(log_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(log_path, filename)
            df = pd.read_csv(file_path)
            logs.append(df)
    
    return pd.concat(logs, ignore_index=True)

def preprocess_logs(logs):
    """
    Preprocess the logs by handling missing data and converting timestamps.
    Args:
        logs (pd.DataFrame): Raw logs
    Returns:
        pd.DataFrame: Cleaned logs
    """
    # Convert timestamp to datetime object
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors='coerce')
    
    # Fill missing values if necessary
    logs.fillna(method='ffill', inplace=True)
    
    return logs
