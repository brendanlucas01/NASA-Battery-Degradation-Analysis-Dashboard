import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table
import glob
import os

# ============================================
# STEP 1: DATA PIPELINE - Load and Process
# ============================================

class BatteryDataPipeline:
    def __init__(self, metadata_path, data_folder):
        self.metadata = pd.read_csv(metadata_path)
        self.metadata['Capacity'] = pd.to_numeric(self.metadata['Capacity'], errors='coerce')
        self.metadata['Re'] = pd.to_numeric(self.metadata['Re'], errors='coerce')
        self.metadata['Rct'] = pd.to_numeric(self.metadata['Rct'], errors='coerce')
        self.data_folder = data_folder
        self.battery_data = {}
        
    def load_cycle_data(self, filename):
        """Load individual cycle CSV file"""
        try:
            filepath = os.path.join(self.data_folder, filename)
            if os.path.exists(filepath):
                return pd.read_csv(filepath)
            return None
        except:
            return None
    
    def calculate_soh(self, battery_id):
        """Calculate SoH for a battery"""
        battery_meta = self.metadata[
            self.metadata['battery_id'] == battery_id
        ].copy()
        
        # Get discharge cycles with valid capacity
        # battery_meta['Capacity'] = pd.to_numeric(battery_meta['Capacity'], errors='coerce')
        discharge_data = battery_meta[
            (battery_meta['type'] == 'discharge') & 
            (battery_meta['Capacity'].notna()) &
            (battery_meta['Capacity'] > 0)
        ].copy()
        
        if len(discharge_data) == 0:
            return None
        
        capacities = discharge_data['Capacity'].values
        test_ids = discharge_data['test_id'].values
        
        # Rated capacity
        rated_capacity = capacities[0]
        
        # Calculate SoH
        soh = (capacities / rated_capacity) * 100
        
        return {
            'battery_id': battery_id,
            'test_ids': test_ids,
            'capacities': capacities,
            'soh': soh,
            'rated_capacity': rated_capacity,
            'cycles': len(capacities),
            'end_of_life_cycle': np.argmax(soh < 80) if np.any(soh < 80) else None
        }
    
    def get_impedance_data(self, battery_id):
        """Extract impedance data"""
        battery_meta = self.metadata[
            self.metadata['battery_id'] == battery_id
        ].copy()
        
        impedance_data = battery_meta[
            (battery_meta['type'] == 'impedance') & 
            (battery_meta['Re'].notna())
        ].copy()
        
        return impedance_data
    
    def process_all_batteries(self):
        """Process all batteries in metadata"""
        batteries = self.metadata['battery_id'].unique()
        results = {}
        
        for battery in batteries:
            soh_data = self.calculate_soh(battery)
            if soh_data:
                results[battery] = soh_data
        
        return results

# ============================================
# STEP 2: DASHBOARD CREATION
# ============================================

# Initialize pipeline
pipeline = BatteryDataPipeline('metadata.csv', '.')
all_battery_data = pipeline.process_all_batteries()

# Initialize Dash app
app = Dash(__name__)

# Get battery list
battery_list = list(all_battery_data.keys())

# Dashboard Layout
app.layout = html.Div([
    html.H1("NASA Battery Health Monitoring Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Select Battery:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='battery-selector',
                options=[{'label': b, 'value': b} for b in battery_list],
                value=battery_list[0] if battery_list else None,
                style={'width': '250px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': 30}),
        
        html.Div([
            html.Label("Compare with:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='compare-battery',
                options=[{'label': b, 'value': b} for b in battery_list],
                multi=True,
                style={'width': '400px'}
            ),
        ], style={'display': 'inline-block'}),
    ], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'marginBottom': 20}),
    
    # Key Metrics Cards
    html.Div(id='metrics-cards', style={'marginBottom': 20}),
    
    # Graphs Row 1
    html.Div([
        html.Div([
            dcc.Graph(id='soh-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='capacity-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    # Graphs Row 2
    html.Div([
        html.Div([
            dcc.Graph(id='impedance-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='degradation-rate-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    # Data Table
    html.H3("Detailed Cycle Data", style={'marginTop': 30}),
    html.Div(id='data-table'),
    
], style={'fontFamily': 'Arial, sans-serif', 'padding': 20})

# ============================================
# CALLBACKS
# ============================================

@app.callback(
    Output('metrics-cards', 'children'),
    Input('battery-selector', 'value')
)
def update_metrics(battery_id):
    if not battery_id or battery_id not in all_battery_data:
        return html.Div("No data available")
    
    data = all_battery_data[battery_id]
    
    cards = html.Div([
        # Card 1: Rated Capacity
        html.Div([
            html.H4("Rated Capacity"),
            html.H2(f"{data['rated_capacity']:.3f} Ah"),
        ], style={
            'display': 'inline-block', 
            'width': '23%', 
            'padding': 20,
            'backgroundColor': '#3498db',
            'color': 'white',
            'borderRadius': 5,
            'marginRight': '2%',
            'textAlign': 'center'
        }),
        
        # Card 2: Total Cycles
        html.Div([
            html.H4("Total Cycles"),
            html.H2(f"{data['cycles']}"),
        ], style={
            'display': 'inline-block', 
            'width': '23%', 
            'padding': 20,
            'backgroundColor': '#2ecc71',
            'color': 'white',
            'borderRadius': 5,
            'marginRight': '2%',
            'textAlign': 'center'
        }),
        
        # Card 3: Current SoH
        html.Div([
            html.H4("Current SoH"),
            html.H2(f"{data['soh'][-1]:.1f}%"),
        ], style={
            'display': 'inline-block', 
            'width': '23%', 
            'padding': 20,
            'backgroundColor': '#e74c3c' if data['soh'][-1] < 80 else '#f39c12',
            'color': 'white',
            'borderRadius': 5,
            'marginRight': '2%',
            'textAlign': 'center'
        }),
        
        # Card 4: Degradation
        # html.Div([
        #     html.H4("Total Degradation"),
        #     html.H2(f"{100-data['soh'][-1]:.1f}%"),
        # ], style={
        #     'display': 'inline-block', 
        #     'width': '23%', 
        #     'padding': 20,
        #     'backgroundColor': '#95a5a6',
        #     'color': 'white',
        #     'borderRadius': 5,
        #     'textAlign': 'center'
        # }),
    ])
    
    return cards

@app.callback(
    Output('soh-graph', 'figure'),
    [Input('battery-selector', 'value'),
     Input('compare-battery', 'value')]
)
def update_soh_graph(battery_id, compare_batteries):
    fig = go.Figure()
    
    # Main battery
    if battery_id and battery_id in all_battery_data:
        data = all_battery_data[battery_id]
        fig.add_trace(go.Scatter(
            x=list(range(len(data['soh']))),
            y=data['soh'],
            mode='lines+markers',
            name=battery_id,
            line=dict(width=3)
        ))
    
    # Comparison batteries
    if compare_batteries:
        for comp_battery in compare_batteries:
            if comp_battery in all_battery_data:
                comp_data = all_battery_data[comp_battery]
                fig.add_trace(go.Scatter(
                    x=list(range(len(comp_data['soh']))),
                    y=comp_data['soh'],
                    mode='lines+markers',
                    name=comp_battery,
                    line=dict(width=2)
                ))
    
    # Add 80% threshold line
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="80% SoH Threshold")
    
    fig.update_layout(
        title="State of Health (SoH) Over Cycles",
        xaxis_title="Cycle Number",
        yaxis_title="SoH (%)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('capacity-graph', 'figure'),
    Input('battery-selector', 'value')
)
def update_capacity_graph(battery_id):
    if not battery_id or battery_id not in all_battery_data:
        return go.Figure()
    
    data = all_battery_data[battery_id]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(data['capacities']))),
        y=data['capacities'],
        mode='lines+markers',
        name='Capacity',
        fill='tozeroy',
        line=dict(color='#3498db', width=2)
    ))
    
    fig.update_layout(
        title=f"Capacity Degradation - {battery_id}",
        xaxis_title="Cycle Number",
        yaxis_title="Capacity (Ah)",
        hovermode='x',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('impedance-graph', 'figure'),
    Input('battery-selector', 'value')
)
def update_impedance_graph(battery_id):
    if not battery_id:
        return go.Figure()
    
    impedance_data = pipeline.get_impedance_data(battery_id)
    
    if len(impedance_data) == 0:
        return go.Figure().add_annotation(
            text="No impedance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    # Re (Electrolyte Resistance)
    if impedance_data['Re'].notna().any():
        fig.add_trace(go.Scatter(
            x=impedance_data['test_id'],
            y=impedance_data['Re'],
            mode='lines+markers',
            name='Re (Electrolyte)',
            line=dict(color='#e74c3c')
        ))
    
    # Rct (Charge Transfer Resistance)
    if impedance_data['Rct'].notna().any():
        fig.add_trace(go.Scatter(
            x=impedance_data['test_id'],
            y=impedance_data['Rct'],
            mode='lines+markers',
            name='Rct (Charge Transfer)',
            line=dict(color='#9b59b6'),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title=f"Impedance Evolution - {battery_id}",
        xaxis_title="Test ID",
        yaxis_title="Re (Ω)",
        yaxis2=dict(
            title="Rct (Ω)",
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        hovermode='x'
    )
    
    return fig

@app.callback(
    Output('degradation-rate-graph', 'figure'),
    Input('battery-selector', 'value')
)
def update_degradation_rate(battery_id):
    if not battery_id or battery_id not in all_battery_data:
        return go.Figure()
    
    data = all_battery_data[battery_id]
    
    # Calculate degradation rate (% per cycle)
    soh_values = data['soh']
    degradation_rate = np.diff(soh_values)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(degradation_rate) + 1)),
        y=degradation_rate,
        name='Degradation Rate',
        marker_color=['red' if x < -1 else 'orange' if x < 0 else 'green' 
                      for x in degradation_rate]
    ))
    
    fig.update_layout(
        title=f"Degradation Rate per Cycle - {battery_id}",
        xaxis_title="Cycle Number",
        yaxis_title="SoH Change (%)",
        template='plotly_white',
        hovermode='x'
    )
    
    return fig

@app.callback(
    Output('data-table', 'children'),
    Input('battery-selector', 'value')
)
def update_table(battery_id):
    if not battery_id or battery_id not in all_battery_data:
        return html.Div("No data available")
    
    data = all_battery_data[battery_id]
    
    # Create table dataframe
    table_df = pd.DataFrame({
        'Cycle': range(len(data['capacities'])),
        'Capacity (Ah)': [f"{c:.4f}" for c in data['capacities']],
        'SoH (%)': [f"{s:.2f}" for s in data['soh']],
        'Status': ['🔴 Critical' if s < 70 else '🟡 Warning' if s < 80 else '🟢 Healthy' 
                   for s in data['soh']]
    })
    
    return dash_table.DataTable(
        data=table_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in table_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': '#3498db',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{Status} contains "Critical"'},
                'backgroundColor': '#ffe6e6',
            },
            {
                'if': {'filter_query': '{Status} contains "Warning"'},
                'backgroundColor': '#fff4e6',
            },
        ],
        page_size=10
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)