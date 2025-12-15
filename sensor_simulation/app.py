import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from simulation import Network
from processing import aggregator_temporal, aggregator_spatial, compress_adaptive, reconstruct_data, calculate_metrics

st.set_page_config(page_title="Sensor Network Aggregation & Compression", layout="wide")

st.title("📡 Sensor Network: Aggregation & Compression Simulation")

# Sidebar Parameters
st.sidebar.header("Simulation Parameters")
num_sensors = st.sidebar.slider("Number of Sensors", 1, 20, 5)
simulation_steps = st.sidebar.slider("Simulation Steps (Data Points)", 10, 500, 100)
agg_window = st.sidebar.slider("Temporal Aggregation Window", 1, 20, 5)
agg_method = st.sidebar.selectbox("Aggregation Method", ["mean", "median", "max", "min"])
spatial_agg = st.sidebar.checkbox("Enable Spatial Aggregation (Cluster)", value=False)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Simulating Distributed Sensor Network..."):
        # 1. Generate Data & Run Distributed Simulation
        network = Network(num_sensors)
        raw_df, sink_df = network.generate_dataset(simulation_steps)
        
        # 2. Process Received Data at Sink
        # The sink_df contains what arrived. 
        # In our simulation, we forwarded raw packets (mostly).
        # Now we apply the "Sink Processing" which includes:
        # - Reconstructing if it was compressed (not fully implemented in transmission yet)
        # - Or applying the Adaptive Compression NOW to show what we COULD do
        #   (since we want to demonstrate the compression logic)
        
        # Let's simulate that the Sink performs the final aggregation/compression 
        # or that we analyze what happened.
        
        # Filter anomalies
        anomalies = sink_df[sink_df['type'] == 'anomaly'] if not sink_df.empty and 'type' in sink_df.columns else pd.DataFrame()
        regular_data = sink_df[sink_df['type'] == 'data'] if not sink_df.empty and 'type' in sink_df.columns else sink_df
        
        # Apply Adaptive Compression to the regular data received
        # This simulates if the network had compressed it perfectly or if we compress at sink
        # To demonstrate the "Adaptive Compression" feature:
        
        target_ratio = st.sidebar.slider("Target Compression Ratio", 1.0, 20.0, 5.0)
        
        # We aggregate first (Temporal)
        agg_df = aggregator_temporal(regular_data, agg_window, agg_method)
        
        # Then Compress Adaptively
        compressed_bytes, orig_size, comp_size, metadata = compress_adaptive(agg_df, target_ratio)
        
        # Then Reconstruct
        reconstructed_df = reconstruct_data(compressed_bytes, metadata)
        
        # Calculate Metrics
        # Compare Original Raw vs Reconstructed
        # Note: reconstructed_df is aggregated, so it has fewer points.
        # calculate_real_mse handles this alignment.
        
        metrics = calculate_metrics(raw_df, reconstructed_df, orig_size, comp_size)
        
        # Display Results
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Compression Ratio", f"{metrics['compression_ratio']:.2f}x")
        col2.metric("Space Saving", f"{metrics['space_saving']:.2f}%")
        col3.metric("Real MSE", f"{metrics['mse_temp']:.4f}")
        col4.metric("Anomalies Detected", f"{len(anomalies)}")
        
        # Charts
        st.subheader("Network & Data Visualization")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Raw Sensor Data (Ground Truth)")
            fig_raw = px.line(raw_df, x='timestamp', y='temperature', color='sensor_id', title="Raw Temperature")
            st.plotly_chart(fig_raw, use_container_width=True)
            
        with col_chart2:
            st.markdown("### Reconstructed Data (at Sink)")
            if not reconstructed_df.empty:
                fig_recon = px.line(reconstructed_df, x='timestamp', y='temperature', color='sensor_id', title="Reconstructed Temperature")
                st.plotly_chart(fig_recon, use_container_width=True)
            else:
                st.info("No data reconstructed.")

        # Anomaly Visualization
        if not anomalies.empty:
            st.subheader("Detected Anomalies")
            st.dataframe(anomalies[['sensor_id', 'timestamp', 'temperature', 'path']])
            
        # Data Inspection
        st.subheader("Data Inspection")
        tab1, tab2, tab3 = st.tabs(["Raw Data", "Sink Received", "Reconstructed"])
        with tab1:
            st.dataframe(raw_df)
        with tab2:
            st.dataframe(sink_df)
        with tab3:
            st.dataframe(reconstructed_df)
            
        st.success(f"Simulation Complete! Network Topology: Tree rooted at Node 0.")

else:
    st.info("Adjust parameters and click 'Run Simulation' to start.")
