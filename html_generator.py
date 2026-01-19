import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_static_dashboard(metadata_path):
    """Generate a standalone HTML dashboard"""
    
    df = pd.read_csv(metadata_path)
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    df['Re'] = pd.to_numeric(df['Re'], errors='coerce')
    df['Rct'] = pd.to_numeric(df['Rct'], errors='coerce')
    batteries = df['battery_id'].unique()[:6]  # First 6 batteries
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('SoH Comparison', 'Capacity Degradation', 
                       'Impedance Growth', 'Cycle Count',
                       'Final SoH Distribution', 'Degradation Rate'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    colors = px.colors.qualitative.Set3
    
    for idx, battery in enumerate(batteries):
        battery_data = df[df['battery_id'] == battery]
        discharge_data = battery_data[
            (battery_data['type'] == 'discharge') &
            (battery_data['Capacity'].notna()) &
            (battery_data['Capacity'] > 0)
        ]
        
        if len(discharge_data) > 0:
            capacities = discharge_data['Capacity'].values
            rated_cap = capacities[0]
            soh = (capacities / rated_cap) * 100
            
            # Plot 1: SoH
            fig.add_trace(
                go.Scatter(x=list(range(len(soh))), y=soh, 
                          name=battery, line=dict(color=colors[idx % len(colors)])),
                row=1, col=1
            )
            
            # Plot 2: Capacity
            fig.add_trace(
                go.Scatter(x=list(range(len(capacities))), y=capacities,
                          name=battery, line=dict(color=colors[idx % len(colors)]),
                          showlegend=False),
                row=1, col=2
            )
    
    # Add 80% threshold
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Cycle", row=1, col=1)
    fig.update_yaxes(title_text="SoH (%)", row=1, col=1)
    fig.update_xaxes(title_text="Cycle", row=1, col=2)
    fig.update_yaxes(title_text="Capacity (Ah)", row=1, col=2)
    
    fig.update_layout(
        height=1200,
        title_text="NASA Battery Dataset - Comprehensive Analysis",
        showlegend=True
    )
    
    # Save to HTML
    fig.write_html('battery_dashboard.html')
    print("Dashboard saved to battery_dashboard.html")
    return fig

# Generate dashboard
create_static_dashboard('metadata.csv')