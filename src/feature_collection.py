# feature_extraction.py

import pandas as pd
import numpy as np

def extract_file_features(logs):
    """
    Extract file-related features from logs (e.g., file modifications, creations, deletions)
    Args:
        logs (pd.DataFrame): Processed logs
    Returns:
        pd.DataFrame: Extracted file features
    """
    features = []
    
    for index, row in logs.iterrows():
        file_modifications = (logs['file_path'] == row['file_path']) & (logs['action'] == 'modify')
        rapid_modifications = file_modifications.sum()  # Count rapid modifications
        
        # Example: Time difference between actions on the same file
        time_diff = logs['timestamp'] - row['timestamp']
        time_diff_seconds = time_diff.total_seconds()
        
        features.append([rapid_modifications, time_diff_seconds.mean()])

    return pd.DataFrame(features, columns=['rapid_modifications', 'avg_time_between_actions'])

def extract_network_features(logs):
    """
    Extract network-related features (e.g., suspicious IPs)
    Args:
        logs (pd.DataFrame): Processed logs
    Returns:
        pd.DataFrame: Extracted network features
    """
    suspicious_ips = ['malicious_ip1', 'malicious_ip2']  # List of known malicious IPs
    logs['suspicious_network_activity'] = logs['destination_ip'].apply(lambda x: x in suspicious_ips)
    
    return logs[['suspicious_network_activity']]

def extract_features(logs):
    """
    Extract all features and combine them into a single DataFrame.
    Args:
        logs (pd.DataFrame): Processed logs
    Returns:
        pd.DataFrame: Combined feature set
    """
    file_features = extract_file_features(logs)
    network_features = extract_network_features(logs)
    return pd.concat([file_features, network_features], axis=1)
