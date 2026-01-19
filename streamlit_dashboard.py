import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="NASA Battery Dashboard",
    page_icon="🔋",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('metadata.csv')
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    df['Re'] = pd.to_numeric(df['Re'], errors='coerce')
    df['Rct'] = pd.to_numeric(df['Rct'], errors='coerce')
    return df

@st.cache_data
def calculate_all_soh(df):
    batteries = df['battery_id'].unique()
    results = {}
    
    for battery in batteries:
        battery_data = df[df['battery_id'] == battery]
        discharge_data = battery_data[
            (battery_data['type'] == 'discharge') & 
            (battery_data['Capacity'].notna()) &
            (battery_data['Capacity'] > 0)
        ]
        
        if len(discharge_data) > 0:
            capacities = discharge_data['Capacity'].values
            rated_capacity = capacities[0]
            soh = (capacities / rated_capacity) * 100
            
            results[battery] = {
                'capacities': capacities,
                'soh': soh,
                'rated_capacity': rated_capacity,
                'cycles': len(capacities)
            }
    
    return results

# Main app
def main():
    st.title("🔋 NASA Battery Health Monitoring Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    battery_data = calculate_all_soh(df)
    
    # Sidebar
    st.sidebar.header("Controls")
    selected_battery = st.sidebar.selectbox(
        "Select Battery",
        options=list(battery_data.keys())
    )
    
    compare_batteries = st.sidebar.multiselect(
        "Compare with",
        options=[b for b in battery_data.keys() if b != selected_battery]
    )
    
    # Metrics
    if selected_battery in battery_data:
        data = battery_data[selected_battery]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Rated Capacity",
                f"{data['rated_capacity']:.3f} Ah"
            )
        
        with col2:
            st.metric(
                "Total Cycles",
                data['cycles']
            )
        
        with col3:
            current_soh = data['soh'][-1]
            st.metric(
                "Current SoH",
                f"{current_soh:.1f}%",
                delta=f"{data['soh'][-1] - data['soh'][0]:.1f}%"
            )
        
        with col4:
            health_status = "🟢 Healthy" if current_soh >= 80 else "🟡 Warning" if current_soh >= 70 else "🔴 Critical"
            st.metric(
                "Status",
                health_status
            )
        
        st.markdown("---")
        
        # Graphs
        col1, col2 = st.columns(2)
        
        with col1:
            # SoH Graph
            fig_soh = go.Figure()
            
            # Main battery
            fig_soh.add_trace(go.Scatter(
                x=list(range(len(data['soh']))),
                y=data['soh'],
                mode='lines+markers',
                name=selected_battery,
                line=dict(width=3)
            ))
            
            # Comparison batteries
            for comp_battery in compare_batteries:
                if comp_battery in battery_data:
                    comp_data = battery_data[comp_battery]
                    fig_soh.add_trace(go.Scatter(
                        x=list(range(len(comp_data['soh']))),
                        y=comp_data['soh'],
                        mode='lines+markers',
                        name=comp_battery
                    ))
            
            fig_soh.add_hline(y=80, line_dash="dash", line_color="red")
            fig_soh.update_layout(
                title="State of Health Over Cycles",
                xaxis_title="Cycle Number",
                yaxis_title="SoH (%)",
                height=400
            )
            st.plotly_chart(fig_soh, use_container_width=True)
        
        with col2:
            # Capacity Graph
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(
                x=list(range(len(data['capacities']))),
                y=data['capacities'],
                mode='lines+markers',
                fill='tozeroy',
                name='Capacity'
            ))
            
            fig_cap.update_layout(
                title="Capacity Degradation",
                xaxis_title="Cycle Number",
                yaxis_title="Capacity (Ah)",
                height=400
            )
            st.plotly_chart(fig_cap, use_container_width=True)
        
        # Impedance Analysis
        st.markdown("### Impedance Analysis")
        battery_meta = df[df['battery_id'] == selected_battery]
        impedance_data = battery_meta[
            (battery_meta['type'] == 'impedance') & 
            (battery_meta['Re'].notna())
        ]
        
        if len(impedance_data) > 0:
            fig_imp = go.Figure()
            
            fig_imp.add_trace(go.Scatter(
                x=impedance_data['test_id'],
                y=impedance_data['Re'],
                mode='lines+markers',
                name='Re (Electrolyte)',
                line=dict(color='red')
            ))
            
            if impedance_data['Rct'].notna().any():
                fig_imp.add_trace(go.Scatter(
                    x=impedance_data['test_id'],
                    y=impedance_data['Rct'],
                    mode='lines+markers',
                    name='Rct (Charge Transfer)',
                    line=dict(color='blue'),
                    yaxis='y2'
                ))
            
            fig_imp.update_layout(
                title="Impedance Evolution",
                xaxis_title="Test ID",
                yaxis_title="Re (Ω)",
                yaxis2=dict(title="Rct (Ω)", overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("No impedance data available for this battery")
        
        # Data Table
        st.markdown("### Cycle-by-Cycle Data")
        table_df = pd.DataFrame({
            'Cycle': range(len(data['capacities'])),
            'Capacity (Ah)': data['capacities'],
            'SoH (%)': data['soh'],
            'Capacity Loss (Ah)': data['rated_capacity'] - data['capacities'],
            'Health Status': ['Critical' if s < 70 else 'Warning' if s < 80 else 'Healthy' 
                             for s in data['soh']]
        })
        
        st.dataframe(table_df, use_container_width=True, height=400)
        
        # Download button
        csv = table_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{selected_battery}_soh_data.csv',
            mime='text/csv'
        )

if __name__ == '__main__':
    main()