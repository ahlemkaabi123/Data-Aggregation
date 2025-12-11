import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from simulation import Network
from processing import aggregator_temporal, aggregator_spatial, compress_data, calculate_metrics

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
    with st.spinner("Simulating Sensor Network..."):
        # 1. Generate Data
        network = Network(num_sensors)
        raw_df = network.generate_dataset(simulation_steps)
        
        # 2. Aggregation
        if spatial_agg:
            # First temporal then spatial, or just spatial?
            # Let's do temporal first for each sensor, then spatial across them?
            # Or just spatial across raw data?
            # The prompt says "Aggregate data in time ... and in space".
            # Let's do temporal aggregation first per sensor to reduce transmission frequency.
            # Then spatial aggregation (simulating a cluster head aggregating from nodes).
            
            temp_agg_df = aggregator_temporal(raw_df, agg_window, agg_method)
            final_agg_df = aggregator_spatial(temp_agg_df, agg_method)
        else:
            final_agg_df = aggregator_temporal(raw_df, agg_window, agg_method)
            
        # 3. Compression
        # Compress the aggregated data
        compressed_bytes, original_size, compressed_size = compress_data(final_agg_df)
        
        # Calculate Metrics
        # For comparison, we need the size of the RAW data if we sent it all
        raw_bytes, raw_size, _ = compress_data(raw_df) # Just to get the size of raw json
        
        # Recalculate compression metrics based on RAW vs FINAL COMPRESSED
        # The 'original_size' returned by compress_data(final_agg_df) is the size of the aggregated data (JSON).
        # But the real saving is Raw Data Size vs Compressed Aggregated Data Size.
        
        metrics = calculate_metrics(raw_df, final_agg_df, raw_size, compressed_size)
        
        # Display Results
        
        # Top Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Compression Ratio", f"{metrics['compression_ratio']:.2f}x")
        col2.metric("Bandwidth Saved", f"{metrics['space_saving']:.2f}%")
        col3.metric("MSE (Temp)", f"{metrics['mse_temp']:.4f}")
        
        # Charts
        st.subheader("Data Visualization")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Raw Sensor Data")
            fig_raw = px.line(raw_df, x='timestamp', y='temperature', color='sensor_id', title="Raw Temperature Data")
            st.plotly_chart(fig_raw, use_container_width=True)
            
        with col_chart2:
            st.markdown("### Aggregated Data")
            if not final_agg_df.empty:
                fig_agg = px.line(final_agg_df, x='timestamp', y='temperature', color='sensor_id', title="Aggregated Temperature Data")
                st.plotly_chart(fig_agg, use_container_width=True)
            else:
                st.info("No aggregated data to display (window size might be too large for dataset)")

        # Data Inspection
        st.subheader("Data Inspection")
        tab1, tab2 = st.tabs(["Raw Data", "Aggregated Data"])
        with tab1:
            st.dataframe(raw_df)
        with tab2:
            st.dataframe(final_agg_df)
            
        st.success(f"Simulation Complete! Original Size: {raw_size} bytes -> Transmitted Size: {compressed_size} bytes")

else:
    st.info("Adjust parameters and click 'Run Simulation' to start.")
