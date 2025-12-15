import pandas as pd
import numpy as np
from simulation import Network
from processing import aggregator_temporal, aggregator_spatial, compress_adaptive, reconstruct_data, calculate_metrics

def test_simulation():
    print("Running Simulation Test...")
    
    # 1. Generate Data
    network = Network(num_sensors=5)
    raw_df, sink_df = network.generate_dataset(steps=20)
    print(f"Generated Raw Data: {raw_df.shape}")
    print(f"Generated Sink Data: {sink_df.shape}")
    assert not raw_df.empty, "Raw data should not be empty"
    # sink_df might be empty if no packets reached sink in 20 steps? 
    # With 5 sensors and 20 steps, and max_hops loop, it should reach.
    assert not sink_df.empty, "Sink data should not be empty"
    
    # 2. Aggregation (Simulated at Sink for test)
    agg_window = 5
    # Filter for regular data
    regular_data = sink_df[sink_df['type'] == 'data']
    if regular_data.empty:
        print("Warning: No regular data received at sink (only anomalies?)")
    else:
        agg_df = aggregator_temporal(regular_data, agg_window, method='mean')
        print(f"Temporal Aggregation: {agg_df.shape}")
        
        # 3. Compression
        compressed_bytes, original_size, compressed_size, metadata = compress_adaptive(agg_df, target_ratio=5.0)
        print(f"Compression: {original_size} -> {compressed_size} bytes")
        assert compressed_size > 0, "Compression failed"
        
        # 4. Reconstruction
        reconstructed_df = reconstruct_data(compressed_bytes, metadata)
        print(f"Reconstruction: {reconstructed_df.shape}")
        assert not reconstructed_df.empty, "Reconstruction failed"
        
        # 5. Metrics
        metrics = calculate_metrics(raw_df, reconstructed_df, original_size, compressed_size)
        print(f"Metrics: {metrics}")
        assert 'compression_ratio' in metrics, "Metrics calculation failed"
        assert 'mse_temp' in metrics, "MSE calculation failed"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_simulation()
