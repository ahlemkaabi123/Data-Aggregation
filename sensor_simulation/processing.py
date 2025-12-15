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

def compress_adaptive(data: pd.DataFrame, target_ratio: float = 5.0) -> Tuple[bytes, int, int, Dict[str, Any]]:
    """
    Compresses data adaptively based on content and target ratio.
    Returns: (compressed_bytes, original_size, compressed_size, metadata)
    """
    if data.empty:
        return b'', 0, 0, {}

    # Metadata to help reconstruction
    metadata = {
        'method': 'zlib',
        'columns': list(data.columns),
        'shape': data.shape,
        'dtypes': data.dtypes.apply(lambda x: str(x)).to_dict()
    }
    
    # 1. Analyze Data
    # Check if temperature is suitable for differential encoding
    # (slowly varying time series)
    use_diff = False
    if 'temperature' in data.columns and len(data) > 1:
        temp_std = data['temperature'].std()
        if temp_std < 2.0: # Low variance, good for diff
            use_diff = True
            
    # 2. Prepare Payload
    payload = data.copy()
    
    if use_diff:
        # Differential encoding for temperature
        # Store first value, then differences
        first_temp = payload['temperature'].iloc[0]
        payload['temperature'] = payload['temperature'].diff().fillna(first_temp)
        metadata['encoding'] = 'differential'
        metadata['first_temp'] = first_temp
    else:
        metadata['encoding'] = 'none'

    # Convert to JSON
    json_str = payload.to_json(orient='records')
    original_bytes = json_str.encode('utf-8')
    
    # 3. Choose Compression Algorithm
    # For humidity (often noisy or int-like), LZ77/Deflate (zlib) is usually good.
    # For very repetitive data, RLE could be better, but zlib handles that well too.
    # Here we simulate a choice.
    
    # If we needed higher compression at cost of CPU, we could use lzma (not implemented here for speed)
    # For this simulation, we stick to zlib but with different levels
    
    if target_ratio > 10:
        level = 9 # Max compression
    elif target_ratio > 5:
        level = 6 # Default
    else:
        level = 1 # Fast
        
    compressed_bytes = zlib.compress(original_bytes, level=level)
    metadata['level'] = level
    
    return compressed_bytes, len(original_bytes), len(compressed_bytes), metadata

def reconstruct_data(compressed_bytes: bytes, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Reconstructs the dataframe from compressed bytes and metadata.
    """
    if not compressed_bytes:
        return pd.DataFrame()
        
    # Decompress
    decompressed_bytes = zlib.decompress(compressed_bytes)
    json_str = decompressed_bytes.decode('utf-8')
    
    df = pd.read_json(json_str)
    
    # Reverse Encoding
    if metadata.get('encoding') == 'differential':
        # Reconstruct cumulative sum
        # The first value was stored as is (fillna), subsequent are diffs
        # But cumsum works if the first value is the start.
        # Our diff() made 2nd row = row2 - row1.
        # So row2_orig = row2_diff + row1_orig.
        # The first row is already the original value.
        df['temperature'] = df['temperature'].cumsum()
        
    return df

def calculate_real_mse(original_df: pd.DataFrame, reconstructed_df: pd.DataFrame) -> float:
    """
    Calculates the Mean Squared Error between original and reconstructed data.
    Matches points by timestamp.
    """
    if original_df.empty or reconstructed_df.empty:
        return 0.0
        
    # Merge on timestamp to align points
    # Round timestamps to ensure matching if there were slight shifts (though there shouldn't be for lossless)
    # If aggregation happened, reconstructed_df will have fewer points.
    
    # If we are comparing Aggregated Reconstructed vs Original Raw
    # We need to map each original point to its nearest aggregated point
    
    # Let's assume we want to measure how well the aggregated data represents the original
    
    # Sort both
    orig = original_df.sort_values('timestamp')
    recon = reconstructed_df.sort_values('timestamp')
    
    # Ensure timestamp types match (convert to float if one is datetime)
    if not pd.api.types.is_float_dtype(orig['timestamp']):
        orig['timestamp'] = orig['timestamp'].astype(float)
    if not pd.api.types.is_float_dtype(recon['timestamp']):
        # If it's datetime, convert to timestamp float
        if pd.api.types.is_datetime64_any_dtype(recon['timestamp']):
             recon['timestamp'] = (recon['timestamp'].astype('int64') // 10**9).astype(float)
        else:
             recon['timestamp'] = recon['timestamp'].astype(float)

    # Use merge_asof to find nearest aggregated point for each original point
    merged = pd.merge_asof(orig, recon, on='timestamp', suffixes=('_orig', '_recon'), direction='nearest')
    
    # Calculate MSE for temperature
    mse = ((merged['temperature_orig'] - merged['temperature_recon']) ** 2).mean()
    
    return mse

def detect_anomalies(df: pd.DataFrame, threshold_z: float = 3.0) -> pd.DataFrame:
    """
    Detects anomalies using Z-score.
    Returns a DataFrame of anomalous rows.
    """
    if df.empty:
        return pd.DataFrame()
        
    anomalies = []
    
    # Simple Z-score per sensor
    for sensor_id, group in df.groupby('sensor_id'):
        if len(group) < 5:
            continue
            
        mean = group['temperature'].mean()
        std = group['temperature'].std()
        
        if std == 0:
            continue
            
        z_scores = np.abs((group['temperature'] - mean) / std)
        sensor_anomalies = group[z_scores > threshold_z]
        anomalies.append(sensor_anomalies)
        
    if anomalies:
        return pd.concat(anomalies)
    return pd.DataFrame()

def calculate_metrics(original_df: pd.DataFrame, reconstructed_df: pd.DataFrame, 
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
        
    # Real MSE
    metrics['mse_temp'] = calculate_real_mse(original_df, reconstructed_df)
    
    return metrics
