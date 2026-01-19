# 🔋 NASA Battery Degradation Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive data analysis dashboard for exploring NASA's Li-ion Battery Aging Dataset, featuring real-time visualization of discharge curves, capacity degradation trends, and predictive analytics for battery health monitoring.

---

## 🎯 Overview

This project provides a comprehensive analytical platform for NASA's Battery Aging Dataset, enabling researchers and engineers to:

- **Visualize** discharge curves and voltage plateaus across multiple charge-discharge cycles
- **Analyze** capacity degradation and State of Health (SoH) trends
- **Compare** battery performance under different test conditions
- **Predict** remaining useful life (RUL) using degradation patterns
- **Export** processed data for machine learning applications

The dashboard is built with **Streamlit** and features an optimized **Snowflake schema** database design for efficient querying of time-series battery data.

---

## ✨ Features

### 📊 Interactive Visualizations
- **Discharge Curve Analysis**: Plot voltage vs. time for individual cycles with zoom/pan capabilities
- **Capacity Fade Tracking**: Monitor capacity degradation across all cycles
- **Multi-Battery Comparison**: Side-by-side analysis of different battery units
- **Statistical Overlays**: Mean, median, and confidence intervals for voltage curves

### 🔍 Advanced Analytics
- **State of Health (SoH) Calculation**: Real-time computation of battery health metrics
- **Cycle-by-Cycle Comparison**: Identify performance changes between consecutive cycles
- **Plateau Voltage Detection**: Automated identification of voltage plateau regions
- **Internal Resistance Estimation**: Derived from voltage drop characteristics

### 📁 Data Management
- **CSV Export**: Download filtered datasets for external analysis
- **Batch Processing**: Analyze multiple batteries simultaneously
- **Data Caching**: Optimized loading with Streamlit's caching mechanism
- **Missing Data Handling**: Robust interpolation for incomplete measurements

---

## 📦 Dataset

### NASA Li-ion Battery Aging Dataset

**Source**: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

**Description**: 
The dataset contains charge-discharge cycle data from Li-ion batteries tested under controlled conditions. Each battery underwent repeated charge-discharge cycles until reaching end-of-life criteria (30% capacity fade).

**Specifications**:
- **Battery Type**: 18650 Li-ion cells
- **Nominal Capacity**: 2.0 Ah
- **Chemistry**: NMC (Nickel Manganese Cobalt)
- **Number of Batteries**: 150+ units
- **Cycles per Battery**: Variable (50-200 cycles)
- **Measurements**: Voltage, Current, Temperature, Capacity
- **Sampling Rate**: 1 Hz

**Data Format**:
```csv
cycle,time,voltage,current,temperature,capacity
0,0.0,4.183,2.000,32.5,0.000
0,1.0,4.175,2.000,32.6,0.001
0,2.0,4.168,2.000,32.7,0.002
```

## Prerequisites
- **Python 3.8 or higher**
- **pip package manager**

Step 1: Clone the Repository
```bash
$ git clone https://github.com/brendanlucas01/NASA-Battery-Degradation-Analysis-Dashboard.git
cd NASA-Battery-Degradation-Analysis-Dashboard
```

Step 2: Create Virtual Environment
# On macOS/Linux
```bash
$ python3 -m venv venv
$ source venv/bin/activate
```
# On Windows
```
$ python -m venv venv
$ venv\Scripts\activate
```

Step 3: Install Dependencies
```bash
$ pip install -r requirements.txt
```
requirements.txt:
```
apache
streamlit==1.28.0
pandas==2.1.0
numpy==1.25.0
matplotlib==3.8.0
plotly==5.17.0
scipy==1.11.0
scikit-learn==1.3.0
```
## 📚 Usage

Run the Streamlit app:
```bash
$ streamlit run advanced_dashboard.py
```

## 📺 Dashboard Components
1. Sidebar Controls
Battery Selector: Dropdown to choose battery unit
Cycle Range: Slider to select cycle range
Visualization Options: Toggle grid lines, legend, tooltips
Export Button: Download current view as CSV/PNG
2. Main Dashboard Panels
Panel A: Discharge Curve Viewer
Interactive Plotly charts showing voltage vs. time with zoom/pan controls and hover tooltips.

Panel B: Capacity Degradation Analysis
- Track capacity fade over cycles with trend lines and statistical analysis.

Panel C: Statistical Summary
Key metrics displayed in real-time:

- Initial Capacity
- Current Capacity
- State of Health (SoH)
- Cycles Completed

Panel D: Comparative Analysis
- Side-by-side battery comparison with normalized overlay plots.

