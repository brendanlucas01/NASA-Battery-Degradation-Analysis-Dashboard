import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import json
from datetime import datetime
import os

# ============================================
# ADVANCED DATA PIPELINE
# ============================================

class AdvancedBatteryPipeline:
    """Complete data pipeline with caching and detailed analysis"""
    
    def __init__(self, metadata_path, data_folder):
        self.metadata_path = metadata_path
        self.data_folder = Path(data_folder)
        self.metadata = None
        self.cache = {}
        self.processed_data = None
        
    def load_metadata(self):
        """Load metadata CSV"""
        try:
            self.metadata = pd.read_csv(self.metadata_path)
            self.metadata['Capacity'] = pd.to_numeric(self.metadata['Capacity'], errors='coerce')
            self.metadata['Re'] = pd.to_numeric(self.metadata['Re'], errors='coerce')
            self.metadata['Rct'] = pd.to_numeric(self.metadata['Rct'], errors='coerce')
            return True
        except Exception as e:
            st.error(f"Error loading metadata: {e}")
            return False
    
    def load_full_cycle_data(self, battery_id):
        """Load all CSV files for a specific battery"""
        if self.metadata is None:
            return []
        
        battery_files = self.metadata[
            self.metadata['battery_id'] == battery_id
        ]['filename'].values
        
        cycle_data = []
        for filename in battery_files:
            filepath = self.data_folder / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    test_info = self.metadata[
                        self.metadata['filename'] == filename
                    ].iloc[0]
                    
                    cycle_data.append({
                        'filename': filename,
                        'test_id': test_info['test_id'],
                        'type': test_info['type'],
                        'ambient_temperature': test_info['ambient_temperature'],
                        'data': df
                    })
                except Exception as e:
                    continue
        
        return cycle_data
    
    def extract_voltage_curves(self, battery_id):
        """Extract discharge voltage curves for analysis"""
        if self.metadata is None:
            return []
        
        battery_meta = self.metadata[
            (self.metadata['battery_id'] == battery_id) &
            (self.metadata['type'] == 'discharge')
        ]
        
        voltage_curves = []
        for _, row in battery_meta.iterrows():
            filepath = self.data_folder / row['filename']
            # print(filepath)
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    # print(df.columns)
                    if 'Voltage_measured' in df.columns:
                        voltage_curves.append({
                            'test_id': row['test_id'],
                            'voltage': df['Voltage_measured'].values,
                            'current': df['Current_measured'].values if 'Current_measured' in df.columns else None,
                            'time': df['Time'].values if 'Time' in df.columns else None,
                            'capacity': row['Capacity'] if 'Capacity' in row else None
                        })
                except Exception as e:
                    continue
        # print(voltage_curves)
        return voltage_curves
    
    def extract_current_profiles(self, battery_id):
        """Extract current profiles for charge/discharge analysis"""
        if self.metadata is None:
            return []
        
        battery_meta = self.metadata[self.metadata['battery_id'] == battery_id]
        
        current_profiles = []
        for _, row in battery_meta.iterrows():
            filepath = self.data_folder / row['filename']
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    if 'Current_measured' in df.columns:
                        current_profiles.append({
                            'test_id': row['test_id'],
                            'type': row['type'],
                            'current': df['Current_measured'].values,
                            'time': df['Time'].values if 'Time' in df.columns else None,
                            'voltage': df['Voltage_measured'].values if 'Voltage_measured' in df.columns else None
                        })
                except:
                    continue
        
        return current_profiles
    
    def calculate_soh(self, battery_id):
        """Calculate State of Health for a battery"""
        if self.metadata is None:
            return None
        
        battery_meta = self.metadata[
            self.metadata['battery_id'] == battery_id
        ].copy()
        
        # Get discharge cycles with valid capacity
        discharge_data = battery_meta[
            (battery_meta['type'] == 'discharge') & 
            (battery_meta['Capacity'].notna()) &
            (battery_meta['Capacity'] > 0)
        ].copy()
        
        if len(discharge_data) == 0:
            return None
        
        discharge_data = discharge_data.sort_values('test_id')
        capacities = discharge_data['Capacity'].values
        test_ids = discharge_data['test_id'].values
        
        # Rated capacity (first cycle)
        rated_capacity = capacities[0]
        
        # Calculate SoH
        soh = (capacities / rated_capacity) * 100
        
        # Calculate degradation rate
        degradation_rates = np.diff(soh)
        
        # Find end of life cycle (SoH < 80%)
        eol_cycle = np.argmax(soh < 80) if np.any(soh < 80) else None
        
        return {
            'battery_id': battery_id,
            'test_ids': test_ids,
            'capacities': capacities,
            'soh': soh,
            'rated_capacity': rated_capacity,
            'final_capacity': capacities[-1],
            'cycles': len(capacities),
            'degradation': 100 - soh[-1],
            'degradation_rates': degradation_rates,
            'avg_degradation_rate': np.mean(degradation_rates),
            'end_of_life_cycle': eol_cycle,
            'end_of_life_reached': eol_cycle is not None
        }
    
    def get_impedance_data(self, battery_id):
        """Extract impedance data for a battery"""
        if self.metadata is None:
            return pd.DataFrame()
        
        battery_meta = self.metadata[
            self.metadata['battery_id'] == battery_id
        ].copy()
        
        impedance_data = battery_meta[
            (battery_meta['type'] == 'impedance') & 
            (battery_meta['Re'].notna())
        ].copy()
        
        return impedance_data
    
    def get_temperature_data(self, battery_id):
        """Extract temperature data"""
        if self.metadata is None:
            return None
        
        battery_meta = self.metadata[
            self.metadata['battery_id'] == battery_id
        ]
        
        return {
            'temperatures': battery_meta['ambient_temperature'].values,
            'test_ids': battery_meta['test_id'].values,
            'avg_temp': battery_meta['ambient_temperature'].mean(),
            'min_temp': battery_meta['ambient_temperature'].min(),
            'max_temp': battery_meta['ambient_temperature'].max()
        }
    
    def process_all_batteries(self):
        """Process all batteries in metadata"""
        if self.metadata is None:
            return {}
        
        batteries = self.metadata['battery_id'].unique()
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, battery in enumerate(batteries):
            status_text.text(f"Processing {battery}... ({idx+1}/{len(batteries)})")
            soh_data = self.calculate_soh(battery)
            if soh_data:
                results[battery] = soh_data
            progress_bar.progress((idx + 1) / len(batteries))
        
        status_text.text("Processing complete!")
        progress_bar.empty()
        status_text.empty()
        
        self.processed_data = results
        return results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.metadata is None:
            return pd.DataFrame()
        
        batteries = self.metadata['battery_id'].unique()
        
        summary_data = []
        for battery in batteries:
            battery_meta = self.metadata[self.metadata['battery_id'] == battery]
            
            discharge_data = battery_meta[
                (battery_meta['type'] == 'discharge') &
                (battery_meta['Capacity'].notna()) &
                (battery_meta['Capacity'] > 0)
            ]
            
            if len(discharge_data) > 0:
                capacities = discharge_data['Capacity'].values
                rated_cap = capacities[0]
                final_cap = capacities[-1]
                soh_final = (final_cap / rated_cap) * 100
                
                # Get temperature info
                avg_temp = battery_meta['ambient_temperature'].mean()
                
                # Get impedance info
                impedance_data = self.get_impedance_data(battery)
                has_impedance = len(impedance_data) > 0
                
                summary_data.append({
                    'Battery ID': battery,
                    'Rated Capacity (Ah)': round(rated_cap, 4),
                    'Final Capacity (Ah)': round(final_cap, 4),
                    'Capacity Loss (Ah)': round(rated_cap - final_cap, 4),
                    'Final SoH (%)': round(soh_final, 2),
                    'Total Cycles': len(discharge_data),
                    'Avg Temperature (°C)': round(avg_temp, 1),
                    'Has Impedance Data': '✓' if has_impedance else '✗',
                    'Status': '🔴 Critical' if soh_final < 70 else '🟡 Warning' if soh_final < 80 else '🟢 Healthy'
                })
        
        return pd.DataFrame(summary_data)
    
    def save_processed_data(self, output_path='processed_battery_data.pkl'):
        """Save processed data for faster loading"""
        if self.processed_data is None:
            self.process_all_batteries()
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.processed_data, f)
        
        return output_path
    
    def load_processed_data(self, input_path='processed_battery_data.pkl'):
        """Load previously processed data"""
        if Path(input_path).exists():
            with open(input_path, 'rb') as f:
                self.processed_data = pickle.load(f)
            return True
        return False
    
    def export_battery_data(self, battery_id, output_format='csv'):
        """Export battery data to various formats"""
        soh_data = self.calculate_soh(battery_id)
        
        if soh_data is None:
            return None
        
        df = pd.DataFrame({
            'Cycle': range(len(soh_data['capacities'])),
            'Test_ID': soh_data['test_ids'],
            'Capacity_Ah': soh_data['capacities'],
            'SoH_Percent': soh_data['soh'],
            'Capacity_Loss_Ah': soh_data['rated_capacity'] - soh_data['capacities'],
        })
        
        if output_format == 'csv':
            return df.to_csv(index=False)
        elif output_format == 'json':
            return df.to_json(orient='records', indent=2)
        elif output_format == 'excel':
            return df.to_excel(index=False)
        
        return None

# ============================================
# STREAMLIT DASHBOARD
# ============================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="NASA Battery Dashboard",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">🔋 NASA Battery Health Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Sidebar - Data Configuration
    st.sidebar.header("⚙️ Configuration")
    
    metadata_path = st.sidebar.text_input(
        "Metadata CSV Path",
        value="metadata.csv",
        help="Path to the metadata.csv file"
    )
    
    data_folder = st.sidebar.text_input(
        "Data Folder Path",
        value=".",
        help="Path to folder containing cycle CSV files"
    )
    
    # Load or reload data
    if st.sidebar.button("🔄 Load/Reload Data", type="primary"):
        with st.spinner("Initializing pipeline..."):
            st.session_state.pipeline = AdvancedBatteryPipeline(metadata_path, data_folder)
            
            if st.session_state.pipeline.load_metadata():
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data loaded successfully!")
                
                # Check for cached processed data
                if st.session_state.pipeline.load_processed_data():
                    st.session_state.processed_data = st.session_state.pipeline.processed_data
                    st.sidebar.info("📦 Loaded cached processed data")
            else:
                st.session_state.data_loaded = False
                st.sidebar.error("❌ Failed to load data")
    
    # Process data button
    if st.session_state.data_loaded and st.sidebar.button("🔬 Process All Batteries"):
        with st.spinner("Processing all batteries..."):
            st.session_state.processed_data = st.session_state.pipeline.process_all_batteries()
            st.session_state.pipeline.save_processed_data()
            st.sidebar.success("✅ Processing complete and cached!")
    
    st.sidebar.markdown("---")
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("👆 Please configure and load data using the sidebar")
        st.markdown("""
        ### Getting Started
        1. Enter the path to your `metadata.csv` file
        2. Enter the path to the folder containing cycle CSV files
        3. Click **Load/Reload Data**
        4. Click **Process All Batteries** to analyze the data
        """)
        return
    
    pipeline = st.session_state.pipeline
    
    # Get battery list
    battery_list = list(pipeline.metadata['battery_id'].unique())
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🔍 Battery Analysis",
        "📈 Comparisons",
        "⚡ Voltage Curves",
        "🌡️ Temperature Analysis",
        "📥 Export Data"
    ])
    
    # ============================================
    # TAB 1: OVERVIEW
    # ============================================
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Batteries",
                len(battery_list),
                help="Number of unique batteries in dataset"
            )
        
        with col2:
            total_tests = len(pipeline.metadata)
            st.metric(
                "Total Tests",
                total_tests,
                help="Total number of test cycles"
            )
        
        with col3:
            discharge_tests = len(pipeline.metadata[pipeline.metadata['type'] == 'discharge'])
            st.metric(
                "Discharge Cycles",
                discharge_tests,
                help="Number of discharge test cycles"
            )
        
        with col4:
            impedance_tests = len(pipeline.metadata[pipeline.metadata['type'] == 'impedance'])
            st.metric(
                "Impedance Tests",
                impedance_tests,
                help="Number of impedance measurements"
            )
        
        st.markdown("---")
        
        # Summary table
        st.subheader("Battery Summary Report")
        summary_df = pipeline.generate_summary_report()
        
        if not summary_df.empty:
            # Color code the status column
            st.dataframe(
                summary_df,
                width='stretch',
                height=400
            )
            
            # Download summary report
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary Report",
                data=csv,
                file_name=f"battery_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SoH Distribution")
            if not summary_df.empty:
                fig = px.histogram(
                    summary_df,
                    x='Final SoH (%)',
                    nbins=20,
                    title="Distribution of Final State of Health",
                    color_discrete_sequence=['#1f77b4']
                )
                fig.add_vline(x=80, line_dash="dash", line_color="red", 
                            annotation_text="80% Threshold")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Cycle Count Distribution")
            if not summary_df.empty:
                fig = px.box(
                    summary_df,
                    y='Total Cycles',
                    title="Distribution of Total Cycles per Battery",
                    color_discrete_sequence=['#2ca02c']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
    
    # ============================================
    # TAB 2: BATTERY ANALYSIS
    # ============================================
    with tab2:
        st.header("Detailed Battery Analysis")
        
        selected_battery = st.selectbox(
            "Select Battery for Analysis",
            options=battery_list,
            key="tab2_battery"
        )
        
        if selected_battery:
            # Calculate or get cached data
            if st.session_state.processed_data and selected_battery in st.session_state.processed_data:
                battery_data = st.session_state.processed_data[selected_battery]
            else:
                battery_data = pipeline.calculate_soh(selected_battery)
            
            if battery_data:
                # Metrics row
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Rated Capacity",
                        f"{battery_data['rated_capacity']:.4f} Ah"
                    )
                
                with col2:
                    st.metric(
                        "Final Capacity",
                        f"{battery_data['final_capacity']:.4f} Ah",
                        delta=f"{battery_data['final_capacity'] - battery_data['rated_capacity']:.4f} Ah"
                    )
                
                with col3:
                    current_soh = battery_data['soh'][-1]
                    st.metric(
                        "Current SoH",
                        f"{current_soh:.2f}%",
                        delta=f"{current_soh - 100:.2f}%"
                    )
                
                with col4:
                    st.metric(
                        "Total Cycles",
                        battery_data['cycles']
                    )
                
                with col5:
                    health_status = "🟢 Healthy" if current_soh >= 80 else "🟡 Warning" if current_soh >= 70 else "🔴 Critical"
                    st.metric(
                        "Status",
                        health_status
                    )
                
                st.markdown("---")
                
                # Graphs
                col1, col2 = st.columns(2)
                
                with col1:
                    # SoH over cycles
                    fig_soh = go.Figure()
                    fig_soh.add_trace(go.Scatter(
                        x=list(range(len(battery_data['soh']))),
                        y=battery_data['soh'],
                        mode='lines+markers',
                        name='SoH',
                        line=dict(width=3, color='#1f77b4'),
                        marker=dict(size=6)
                    ))
                    
                    fig_soh.add_hline(
                        y=80,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="80% Threshold"
                    )
                    
                    if battery_data['end_of_life_reached']:
                        fig_soh.add_vline(
                            x=battery_data['end_of_life_cycle'],
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="End of Life"
                        )
                    
                    fig_soh.update_layout(
                        title="State of Health Over Cycles",
                        xaxis_title="Cycle Number",
                        yaxis_title="SoH (%)",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig_soh, width='stretch')
                
                with col2:
                    # Capacity degradation
                    fig_cap = go.Figure()
                    fig_cap.add_trace(go.Scatter(
                        x=list(range(len(battery_data['capacities']))),
                        y=battery_data['capacities'],
                        mode='lines+markers',
                        name='Capacity',
                        fill='tozeroy',
                        line=dict(width=3, color='#2ca02c'),
                        marker=dict(size=6)
                    ))
                    
                    fig_cap.update_layout(
                        title="Capacity Degradation",
                        xaxis_title="Cycle Number",
                        yaxis_title="Capacity (Ah)",
                        hovermode='x',
                        height=400
                    )
                    st.plotly_chart(fig_cap, width='stretch')
                
                # Degradation rate analysis
                st.subheader("Degradation Rate Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_deg = go.Figure()
                    fig_deg.add_trace(go.Bar(
                        x=list(range(1, len(battery_data['degradation_rates']) + 1)),
                        y=battery_data['degradation_rates'],
                        name='Degradation Rate',
                        marker_color=['#d62728' if x < -1 else '#ff7f0e' if x < 0 else '#2ca02c' 
                                     for x in battery_data['degradation_rates']]
                    ))
                    
                    fig_deg.update_layout(
                        title="Cycle-to-Cycle Degradation Rate",
                        xaxis_title="Cycle Number",
                        yaxis_title="SoH Change (%)",
                        height=400
                    )
                    st.plotly_chart(fig_deg, width='stretch')
                
                with col2:
                    # Moving average of degradation
                    window_size = min(10, len(battery_data['degradation_rates']))
                    if len(battery_data['degradation_rates']) > window_size:
                        moving_avg = np.convolve(
                            battery_data['degradation_rates'],
                            np.ones(window_size)/window_size,
                            mode='valid'
                        )
                        
                        fig_ma = go.Figure()
                        fig_ma.add_trace(go.Scatter(
                            x=list(range(window_size, len(battery_data['degradation_rates']) + 1)),
                            y=moving_avg,
                            mode='lines',
                            name=f'{window_size}-Cycle Moving Average',
                            line=dict(width=3, color='#9467bd')
                        ))
                        
                        fig_ma.update_layout(
                            title=f"Degradation Rate Trend ({window_size}-Cycle MA)",
                            xaxis_title="Cycle Number",
                            yaxis_title="Avg SoH Change (%)",
                            height=400
                        )
                        st.plotly_chart(fig_ma, width='stretch')
                
                # Impedance analysis
                st.subheader("Impedance Analysis")
                impedance_data = pipeline.get_impedance_data(selected_battery)
                
                if not impedance_data.empty:
                    fig_imp = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Electrolyte Resistance (Re)", "Charge Transfer Resistance (Rct)")
                    )
                    
                    # Re plot
                    if impedance_data['Re'].notna().any():
                        fig_imp.add_trace(
                            go.Scatter(
                                x=impedance_data['test_id'],
                                y=impedance_data['Re'],
                                mode='lines+markers',
                                name='Re',
                                line=dict(color='#d62728')
                            ),
                            row=1, col=1
                        )
                    
                    # Rct plot
                    if impedance_data['Rct'].notna().any():
                        fig_imp.add_trace(
                            go.Scatter(
                                x=impedance_data['test_id'],
                                y=impedance_data['Rct'],
                                mode='lines+markers',
                                name='Rct',
                                line=dict(color='#9467bd')
                            ),
                            row=1, col=2
                        )
                    
                    fig_imp.update_xaxes(title_text="Test ID", row=1, col=1)
                    fig_imp.update_xaxes(title_text="Test ID", row=1, col=2)
                    fig_imp.update_yaxes(title_text="Resistance (Ω)", row=1, col=1)
                    fig_imp.update_yaxes(title_text="Resistance (Ω)", row=1, col=2)
                    
                    fig_imp.update_layout(
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_imp, width='stretch')
                else:
                    st.info("No impedance data available for this battery")
                
                # Data table
                st.subheader("Cycle-by-Cycle Data")
                table_df = pd.DataFrame({
                    'Cycle': range(len(battery_data['capacities'])),
                    'Test ID': battery_data['test_ids'],
                    'Capacity (Ah)': battery_data['capacities'],
                    'SoH (%)': battery_data['soh'],
                    'Capacity Loss (Ah)': battery_data['rated_capacity'] - battery_data['capacities'],
                    'Health Status': ['🔴 Critical' if s < 70 else '🟡 Warning' if s < 80 else '🟢 Healthy' 
                                     for s in battery_data['soh']]
                })
                
                st.dataframe(table_df, width='stretch', height=400)
    
    # ============================================
    # TAB 3: COMPARISONS
    # ============================================
    with tab3:
        st.header("Multi-Battery Comparison")
        
        compare_batteries = st.multiselect(
            "Select Batteries to Compare (up to 6)",
            options=battery_list,
            default=battery_list[:min(3, len(battery_list))],
            max_selections=6
        )
        
        if compare_batteries:
            # SoH Comparison
            st.subheader("State of Health Comparison")
            
            fig_comp = go.Figure()
            
            colors = px.colors.qualitative.Set3
            
            for idx, battery in enumerate(compare_batteries):
                if st.session_state.processed_data and battery in st.session_state.processed_data:
                    data = st.session_state.processed_data[battery]
                else:
                    data = pipeline.calculate_soh(battery)
                
                if data:
                    fig_comp.add_trace(go.Scatter(
                        x=list(range(len(data['soh']))),
                        y=data['soh'],
                        mode='lines+markers',
                        name=battery,
                        line=dict(width=2, color=colors[idx % len(colors)]),
                        marker=dict(size=4)
                    ))
            
            fig_comp.add_hline(
                y=80,
                line_dash="dash",
                line_color="red",
                annotation_text="80% Threshold"
            )
            
            fig_comp.update_layout(
                title="SoH Comparison Across Selected Batteries",
                xaxis_title="Cycle Number",
                yaxis_title="SoH (%)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_comp, width='stretch')
            
            # Capacity comparison
            st.subheader("Capacity Comparison")
            
            fig_cap_comp = go.Figure()
            
            for idx, battery in enumerate(compare_batteries):
                if st.session_state.processed_data and battery in st.session_state.processed_data:
                    data = st.session_state.processed_data[battery]
                else:
                    data = pipeline.calculate_soh(battery)
                
                if data:
                    fig_cap_comp.add_trace(go.Scatter(
                        x=list(range(len(data['capacities']))),
                        y=data['capacities'],
                        mode='lines+markers',
                        name=battery,
                        line=dict(width=2, color=colors[idx % len(colors)]),
                        marker=dict(size=4)
                    ))
            
            fig_cap_comp.update_layout(
                title="Capacity Degradation Comparison",
                xaxis_title="Cycle Number",
                yaxis_title="Capacity (Ah)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_cap_comp, width='stretch')
            
            # Comparison metrics
            st.subheader("Comparative Metrics")
            
            comparison_data = []
            for battery in compare_batteries:
                if st.session_state.processed_data and battery in st.session_state.processed_data:
                    data = st.session_state.processed_data[battery]
                else:
                    data = pipeline.calculate_soh(battery)
                
                if data:
                    comparison_data.append({
                        'Battery': battery,
                        'Rated Capacity (Ah)': round(data['rated_capacity'], 4),
                        'Final SoH (%)': round(data['soh'][-1], 2),
                        'Total Degradation (%)': round(data['degradation'], 2),
                        'Avg Degradation Rate (%/cycle)': round(data['avg_degradation_rate'], 4),
                        'Total Cycles': data['cycles'],
                        'End of Life': '✓' if data['end_of_life_reached'] else '✗'
                    })
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, width='stretch')
    
    # ============================================
    # TAB 4: VOLTAGE CURVES
    # ============================================
    with tab4:
        st.header("Discharge Voltage Curves")
        
        voltage_battery = st.selectbox(
            "Select Battery",
            options=battery_list,
            key="voltage_battery"
        )
        
        if voltage_battery:
            with st.spinner("Loading voltage curve data..."):
                voltage_curves = pipeline.extract_voltage_curves(voltage_battery)
            
            if voltage_curves:
                st.success(f"Loaded {len(voltage_curves)} discharge cycles")
                
                # Cycle selection
                cycle_indices = list(range(len(voltage_curves)))
                selected_cycles = st.multiselect(
                    "Select Cycles to Display",
                    options=cycle_indices,
                    default=cycle_indices[:min(5, len(cycle_indices))],
                    format_func=lambda x: f"Cycle {x} (Test {voltage_curves[x]['test_id']})"
                )
                
                if selected_cycles:
                    fig_volt = go.Figure()
                    
                    colors = px.colors.sequential.Viridis
                    
                    for idx in selected_cycles:
                        curve = voltage_curves[idx]
                        color_idx = int((idx / len(voltage_curves)) * len(colors))
                        
                        # Use time if available, otherwise use index
                        if curve['time'] is not None:
                            x_data = curve['time']
                            x_label = "Time (s)"
                        else:
                            x_data = list(range(len(curve['voltage'])))
                            x_label = "Sample Index"
                        
                        fig_volt.add_trace(go.Scatter(
                            x=x_data,
                            y=curve['voltage'],
                            mode='lines',
                            name=f"Cycle {idx}",
                            line=dict(width=2, color=colors[color_idx % len(colors)])
                        ))
                    
                    fig_volt.update_layout(
                        title=f"Discharge Voltage Curves - {voltage_battery}",
                        xaxis_title=x_label,
                        yaxis_title="Voltage (V)",
                        hovermode='x unified',
                        height=600
                    )
                    st.plotly_chart(fig_volt, width='stretch')
                    
                    # Voltage statistics
                    st.subheader("Voltage Curve Statistics")
                    
                    voltage_stats = []
                    for idx in selected_cycles:
                        curve = voltage_curves[idx]
                        voltage_stats.append({
                            'Cycle': idx,
                            'Test ID': curve['test_id'],
                            'Min Voltage (V)': round(np.min(curve['voltage']), 4),
                            'Max Voltage (V)': round(np.max(curve['voltage']), 4),
                            'Avg Voltage (V)': round(np.mean(curve['voltage']), 4),
                            'Voltage Range (V)': round(np.max(curve['voltage']) - np.min(curve['voltage']), 4),
                            'Capacity (Ah)': round(curve['capacity'], 4) if curve['capacity'] else 'N/A'
                        })
                    
                    stats_df = pd.DataFrame(voltage_stats)
                    st.dataframe(stats_df, width='stretch')
            else:
                st.warning("No voltage curve data available for this battery")
    
    # ============================================
    # TAB 5: TEMPERATURE ANALYSIS
    # ============================================
    with tab5:
        st.header("Temperature Analysis")
        
        temp_battery = st.selectbox(
            "Select Battery",
            options=battery_list,
            key="temp_battery"
        )
        
        if temp_battery:
            temp_data = pipeline.get_temperature_data(temp_battery)
            
            if temp_data:
                # Temperature metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Temperature", f"{temp_data['avg_temp']:.1f}°C")
                
                with col2:
                    st.metric("Min Temperature", f"{temp_data['min_temp']:.1f}°C")
                
                with col3:
                    st.metric("Max Temperature", f"{temp_data['max_temp']:.1f}°C")
                
                # Temperature over time
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=temp_data['test_ids'],
                    y=temp_data['temperatures'],
                    mode='lines+markers',
                    name='Ambient Temperature',
                    line=dict(width=2, color='#ff7f0e'),
                    marker=dict(size=6)
                ))
                
                fig_temp.add_hline(
                    y=temp_data['avg_temp'],
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Average: {temp_data['avg_temp']:.1f}°C"
                )
                
                fig_temp.update_layout(
                    title=f"Temperature Profile - {temp_battery}",
                    xaxis_title="Test ID",
                    yaxis_title="Temperature (°C)",
                    height=500
                )
                st.plotly_chart(fig_temp, width='stretch')
                
                # Temperature distribution
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=temp_data['temperatures'],
                    nbinsx=20,
                    name='Temperature Distribution',
                    marker_color='#ff7f0e'
                ))
                
                fig_hist.update_layout(
                    title="Temperature Distribution",
                    xaxis_title="Temperature (°C)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_hist, width='stretch')
    
    # ============================================
    # TAB 6: EXPORT DATA
    # ============================================
    with tab6:
        st.header("Export Data")
        
        export_battery = st.selectbox(
            "Select Battery to Export",
            options=battery_list,
            key="export_battery"
        )
        
        export_format = st.radio(
            "Select Export Format",
            options=["CSV", "JSON"],
            horizontal=True
        )
        
        if st.button("Generate Export File", type="primary"):
            if export_battery:
                with st.spinner("Generating export file..."):
                    if export_format == "CSV":
                        export_data = pipeline.export_battery_data(export_battery, 'csv')
                        file_ext = "csv"
                        mime_type = "text/csv"
                    else:
                        export_data = pipeline.export_battery_data(export_battery, 'json')
                        file_ext = "json"
                        mime_type = "application/json"
                    
                    if export_data:
                        st.download_button(
                            label=f"📥 Download {export_battery} Data ({export_format})",
                            data=export_data,
                            file_name=f"{export_battery}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime_type
                        )
                        st.success("Export file generated successfully!")
        
        st.markdown("---")
        
        # Bulk export
        st.subheader("Bulk Export Options")
        
        if st.button("Export All Processed Data"):
            if st.session_state.processed_data:
                # Create comprehensive export
                all_data = []
                for battery_id, data in st.session_state.processed_data.items():
                    for i in range(len(data['capacities'])):
                        all_data.append({
                            'Battery_ID': battery_id,
                            'Cycle': i,
                            'Test_ID': data['test_ids'][i],
                            'Capacity_Ah': data['capacities'][i],
                            'SoH_Percent': data['soh'][i],
                            'Rated_Capacity_Ah': data['rated_capacity']
                        })
                
                export_df = pd.DataFrame(all_data)
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Complete Dataset (CSV)",
                    data=csv_data,
                    file_name=f"all_batteries_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Please process batteries first before bulk export")
        
        # Export summary report
        if st.button("Export Summary Report"):
            summary_df = pipeline.generate_summary_report()
            if not summary_df.empty:
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary Report (CSV)",
                    data=csv_data,
                    file_name=f"battery_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ============================================
# RUN THE APP
# ============================================

if __name__ == "__main__":
    main()