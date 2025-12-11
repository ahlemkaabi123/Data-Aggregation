import zlib
import pandas as pd
import numpy as np
import json
from typing import Tuple, Dict, Any

def aggregator_temporal(df: pd.DataFrame, window_size: int, method: str = 'mean') -> pd.DataFrame:
    """
    Aggregates data over time windows for each sensor.
    Assumes df has 'sensor_id', 'timestamp', 'temperature', 'humidity'.
    """
    if df.empty:
        return df

    # Ensure timestamp is sorted
    df = df.sort_values('timestamp')
    
    # Group by sensor and apply rolling window aggregation
    # Since we want to reduce data, we can resample or group by a custom index
    # Here we simulate windowing by grouping every N rows per sensor
    
    aggregated_rows = []
    
    for sensor_id, group in df.groupby('sensor_id'):
        # Create a grouping key for every 'window_size' rows
        group['group_key'] = np.arange(len(group)) // window_size
        
        if method == 'mean':
            agg = group.groupby('group_key')[['temperature', 'humidity']].mean()
        elif method == 'median':
            agg = group.groupby('group_key')[['temperature', 'humidity']].median()
        elif method == 'max':
            agg = group.groupby('group_key')[['temperature', 'humidity']].max()
        elif method == 'min':
            agg = group.groupby('group_key')[['temperature', 'humidity']].min()
        else:
            agg = group.groupby('group_key')[['temperature', 'humidity']].mean()
            
        # Take the last timestamp of the window as the timestamp for the aggregated point
        agg['timestamp'] = group.groupby('group_key')['timestamp'].max()
        agg['sensor_id'] = sensor_id
        agg['count'] = group.groupby('group_key')['timestamp'].count()
        
        aggregated_rows.append(agg)
        
    if not aggregated_rows:
        return pd.DataFrame()
        
    result = pd.concat(aggregated_rows).reset_index(drop=True)
    return result

def aggregator_spatial(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """
    Aggregates data across all sensors at each timestamp (or close timestamps).
    For simplicity, we assume synchronized timestamps or bin them.
    """
    if df.empty:
        return df
        
    # Bin timestamps to nearest second to align sensors
    df['time_bin'] = df['timestamp'].round(0)
    
    if method == 'mean':
        agg = df.groupby('time_bin')[['temperature', 'humidity']].mean()
    elif method == 'median':
        agg = df.groupby('time_bin')[['temperature', 'humidity']].median()
    else:
        agg = df.groupby('time_bin')[['temperature', 'humidity']].mean()
        
    agg = agg.reset_index().rename(columns={'time_bin': 'timestamp'})
    agg['sensor_id'] = -1 # Indicates spatial aggregation (cluster head)
    
    return agg

def compress_data(data: pd.DataFrame) -> Tuple[bytes, int, int]:
    """
    Compresses the dataframe using zlib.
    Returns: (compressed_bytes, original_size, compressed_size)
    """
    # Convert to JSON string first as a common payload format
    json_str = data.to_json(orient='records')
    original_bytes = json_str.encode('utf-8')
    compressed_bytes = zlib.compress(original_bytes)
    
    return compressed_bytes, len(original_bytes), len(compressed_bytes)

def calculate_metrics(original_df: pd.DataFrame, aggregated_df: pd.DataFrame, 
                     original_size: int, compressed_size: int) -> Dict[str, float]:
    """
    Calculates compression metrics and error.
    """
    metrics = {}
    
    # Compression Ratio
    if compressed_size > 0:
        metrics['compression_ratio'] = original_size / compressed_size
        metrics['space_saving'] = (1 - (compressed_size / original_size)) * 100
    else:
        metrics['compression_ratio'] = 1.0
        metrics['space_saving'] = 0.0
        
    # MSE (Mean Squared Error) - tricky if aggregation reduced row count
    # We compare the aggregated values to the original values they represent.
    # For simplicity in this simulation, we can just compare the means if the shapes differ significantly,
    # or we can try to map back.
    # A simple approach: Compare the mean of the original dataset vs mean of aggregated dataset
    
    orig_mean_temp = original_df['temperature'].mean()
    agg_mean_temp = aggregated_df['temperature'].mean()
    
    metrics['mse_temp'] = (orig_mean_temp - agg_mean_temp) ** 2
    
    return metrics
