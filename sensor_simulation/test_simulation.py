import pandas as pd
import numpy as np
from simulation import Network
from processing import aggregator_temporal, aggregator_spatial, compress_data, calculate_metrics

def test_simulation():
    print("Running Simulation Test...")
    
    # 1. Generate Data
    network = Network(num_sensors=5)
    raw_df = network.generate_dataset(steps=100)
    print(f"Generated Raw Data: {raw_df.shape}")
    assert not raw_df.empty, "Raw data should not be empty"
    
    # 2. Aggregation
    agg_window = 5
    temp_agg_df = aggregator_temporal(raw_df, agg_window, method='mean')
    print(f"Temporal Aggregation: {temp_agg_df.shape}")
    assert not temp_agg_df.empty, "Temporal aggregation failed"
    
    final_agg_df = aggregator_spatial(temp_agg_df, method='mean')
    print(f"Spatial Aggregation: {final_agg_df.shape}")
    assert not final_agg_df.empty, "Spatial aggregation failed"
    
    # 3. Compression
    compressed_bytes, original_size, compressed_size = compress_data(final_agg_df)
    print(f"Compression: {original_size} -> {compressed_size} bytes")
    assert compressed_size > 0, "Compression failed"
    
    # 4. Metrics
    raw_bytes, raw_size, _ = compress_data(raw_df)
    metrics = calculate_metrics(raw_df, final_agg_df, raw_size, compressed_size)
    print(f"Metrics: {metrics}")
    assert 'compression_ratio' in metrics, "Metrics calculation failed"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_simulation()
