import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="ManuSense - Smart Manufacturing Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved text sizes and UI
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Enhanced metrics styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .stMetric label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #495057 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 16px !important;
    }
    
    /* Headers styling */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        font-size: 42px !important;
        margin-bottom: 20px !important;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        font-size: 32px !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
    }
    
    h3 {
        color: #34495e;
        font-weight: 600;
        font-size: 24px !important;
        margin-bottom: 12px !important;
    }
    
    /* Alert boxes with better typography */
    .alert-box {
        padding: 18px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .alert-critical {
        background-color: #ffebee;
        border-left: 6px solid #f44336;
        color: #c62828;
    }
    
    .alert-warning {
        background-color: #fff3e0;
        border-left: 6px solid #ff9800;
        color: #e65100;
    }
    
    .alert-normal {
        background-color: #e8f5e9;
        border-left: 6px solid #4caf50;
        color: #2e7d32;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 16px !important;
        font-weight: 500 !important;
        padding: 8px 0;
    }
    
    section[data-testid="stSidebar"] h3 {
        font-size: 20px !important;
        color: #1f77b4;
    }
    
    /* Button styling */
    .stButton button {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
    }
    
    /* Select box and input styling */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
    }
    
    /* Info/success/warning/error boxes */
    .stAlert {
        font-size: 16px !important;
        padding: 16px !important;
    }
    
    /* General text sizing */
    p, li, span {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Load models and data
@st.cache_resource
def load_models():
    """Load trained ML models"""
    models = {}
    model_files = {
        'quality': 'quality_model.pkl',
        'maintenance': 'maintenance_model.pkl',
        'energy': 'energy_model.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            models[model_name] = joblib.load(filepath)
    
    return models

@st.cache_data
def load_sample_data():
    """Load or generate sample manufacturing data"""
    # Generate sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='h')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'machine_id': np.random.choice(['M001', 'M002', 'M003', 'M004'], 1000),
        'temperature': np.random.normal(75, 5, 1000),
        'pressure': np.random.normal(100, 10, 1000),
        'vibration': np.random.normal(0.5, 0.1, 1000),
        'power_consumption': np.random.normal(150, 20, 1000),
        'production_count': np.random.randint(80, 120, 1000),
        'defect_count': np.random.randint(0, 5, 1000),
        'operator_id': np.random.choice(['OP001', 'OP002', 'OP003'], 1000)
    })
    
    # Add quality score
    data['quality_score'] = 100 - (data['defect_count'] / data['production_count'] * 100)
    
    # Add anomaly flags
    data['is_anomaly'] = (
        (data['temperature'] > 85) | 
        (data['pressure'] > 120) | 
        (data['vibration'] > 0.7)
    ).astype(int)
    
    return data

def login_page():
    """Display login page"""
    st.title("🏭 ManuSense Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Welcome Back!")
        st.markdown("Please sign in to access the manufacturing analytics dashboard.")
        st.markdown("---")
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        st.markdown("")  # Add spacing
        
        if st.button("Login", use_container_width=True, type="primary"):
            # Simple authentication (in production, use proper authentication)
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("⚠️ Please enter both username and password")
        
        st.markdown("---")
        st.info("**Demo Credentials:**\n\n👤 Username: `demo`\n\n🔑 Password: `demo`")
        
        st.markdown("")
        st.markdown("""
            <div style='text-align: center; color: #6c757d; font-size: 14px; margin-top: 30px;'>
                <p>ManuSense v1.0 - Smart Manufacturing Analytics Platform</p>
                <p>Powered by AI & Machine Learning</p>
            </div>
        """, unsafe_allow_html=True)

def logout():
    """Handle logout"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

def display_kpis(data):
    """Display key performance indicators"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate KPIs
    recent_data = data[data['timestamp'] > (datetime.now() - timedelta(hours=24))]
    
    total_production = recent_data['production_count'].sum()
    total_defects = recent_data['defect_count'].sum()
    avg_quality = recent_data['quality_score'].mean()
    avg_energy = recent_data['power_consumption'].mean()
    anomaly_count = recent_data['is_anomaly'].sum()
    
    with col1:
        st.metric(
            label="Total Production (24h)",
            value=f"{total_production:,}",
            delta=f"+{np.random.randint(5, 15)}%"
        )
    
    with col2:
        st.metric(
            label="Defect Rate",
            value=f"{(total_defects/total_production*100):.2f}%",
            delta=f"-{np.random.randint(1, 5)}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Avg Quality Score",
            value=f"{avg_quality:.1f}%",
            delta=f"+{np.random.randint(1, 3)}%"
        )
    
    with col4:
        st.metric(
            label="Energy Consumption",
            value=f"{avg_energy:.0f} kWh",
            delta=f"-{np.random.randint(2, 8)}%",
            delta_color="inverse"
        )
    
    with col5:
        st.metric(
            label="Anomalies Detected",
            value=anomaly_count,
            delta=f"{np.random.randint(-5, 5)}"
        )

def production_overview_tab(data):
    """Display production overview"""
    st.header("📊 Production Overview")
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        time_range = st.selectbox(
            "Select Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"]
        )
    
    # Filter data based on time range
    if time_range == "Last 24 Hours":
        filtered_data = data[data['timestamp'] > (datetime.now() - timedelta(hours=24))]
    elif time_range == "Last 7 Days":
        filtered_data = data[data['timestamp'] > (datetime.now() - timedelta(days=7))]
    elif time_range == "Last 30 Days":
        filtered_data = data[data['timestamp'] > (datetime.now() - timedelta(days=30))]
    else:
        filtered_data = data
    
    # Production trend chart
    st.subheader("Production Trend")
    hourly_production = filtered_data.groupby(filtered_data['timestamp'].dt.floor('h')).agg({
        'production_count': 'sum',
        'defect_count': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_production['timestamp'],
        y=hourly_production['production_count'],
        mode='lines+markers',
        name='Production',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=hourly_production['timestamp'],
        y=hourly_production['defect_count'],
        mode='lines+markers',
        name='Defects',
        line=dict(color='#ff7f0e', width=2)
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Count",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Machine performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Machine Performance")
        machine_stats = filtered_data.groupby('machine_id').agg({
            'production_count': 'sum',
            'quality_score': 'mean'
        }).reset_index()
        
        fig = px.bar(
            machine_stats,
            x='machine_id',
            y='production_count',
            color='quality_score',
            color_continuous_scale='RdYlGn',
            labels={'production_count': 'Total Production', 'quality_score': 'Avg Quality Score'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Operator Performance")
        operator_stats = filtered_data.groupby('operator_id').agg({
            'production_count': 'sum',
            'defect_count': 'sum'
        }).reset_index()
        operator_stats['defect_rate'] = (operator_stats['defect_count'] / operator_stats['production_count'] * 100)
        
        fig = px.bar(
            operator_stats,
            x='operator_id',
            y='production_count',
            color='defect_rate',
            color_continuous_scale='RdYlGn_r',
            labels={'production_count': 'Total Production', 'defect_rate': 'Defect Rate (%)'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

def quality_monitoring_tab(data):
    """Display quality monitoring dashboard"""
    st.header("🎯 Quality Monitoring")
    
    recent_data = data[data['timestamp'] > (datetime.now() - timedelta(hours=24))]
    
    # Quality trend
    st.subheader("Quality Score Trend")
    hourly_quality = recent_data.groupby(recent_data['timestamp'].dt.floor('h')).agg({
        'quality_score': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_quality['timestamp'],
        y=hourly_quality['quality_score'],
        mode='lines',
        name='Quality Score',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=2)
    ))
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target: 95%")
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Quality Score (%)",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Defect analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Defect Distribution by Machine")
        defect_by_machine = recent_data.groupby('machine_id')['defect_count'].sum().reset_index()
        fig = px.pie(
            defect_by_machine,
            values='defect_count',
            names='machine_id',
            title='Defect Distribution'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Quality Score Distribution")
        fig = px.histogram(
            recent_data,
            x='quality_score',
            nbins=20,
            title='Quality Score Frequency',
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quality alerts
    st.subheader("Quality Alerts")
    low_quality = recent_data[recent_data['quality_score'] < 90].tail(5)
    
    if len(low_quality) > 0:
        for idx, row in low_quality.iterrows():
            alert_type = "alert-critical" if row['quality_score'] < 85 else "alert-warning"
            st.markdown(
                f"""
                <div class="alert-box {alert_type}">
                    <strong>Machine {row['machine_id']}</strong> - Quality Score: {row['quality_score']:.1f}% 
                    | Defects: {row['defect_count']} | Time: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No quality issues detected in the last 24 hours")

def predictive_maintenance_tab(data):
    """Display predictive maintenance dashboard"""
    st.header("🔧 Predictive Maintenance")
    
    # Machine health status
    st.subheader("Machine Health Status")
    
    machines = data['machine_id'].unique()
    recent_data = data[data['timestamp'] > (datetime.now() - timedelta(hours=24))]
    
    cols = st.columns(len(machines))
    
    for idx, machine in enumerate(machines):
        machine_data = recent_data[recent_data['machine_id'] == machine]
        
        # Calculate health score (simplified)
        temp_score = 100 - abs(machine_data['temperature'].mean() - 75) * 2
        pressure_score = 100 - abs(machine_data['pressure'].mean() - 100) * 0.5
        vibration_score = 100 - (machine_data['vibration'].mean() - 0.5) * 100
        health_score = np.mean([temp_score, pressure_score, vibration_score])
        
        with cols[idx]:
            if health_score > 80:
                status_color = "🟢"
                status_text = "Healthy"
            elif health_score > 60:
                status_color = "🟡"
                status_text = "Warning"
            else:
                status_color = "🔴"
                status_text = "Critical"
            
            st.metric(
                label=f"{status_color} {machine}",
                value=f"{health_score:.0f}%",
                delta=status_text
            )
    
    # Sensor readings
    st.subheader("Real-time Sensor Readings")
    
    selected_machine = st.selectbox("Select Machine", machines)
    machine_data = recent_data[recent_data['machine_id'] == selected_machine]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Temperature (°F)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=machine_data['timestamp'],
            y=machine_data['temperature'],
            mode='lines',
            line=dict(color='#e74c3c', width=2)
        ))
        fig.add_hrect(y0=70, y1=80, fillcolor="green", opacity=0.1, line_width=0)
        fig.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pressure (PSI)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=machine_data['timestamp'],
            y=machine_data['pressure'],
            mode='lines',
            line=dict(color='#3498db', width=2)
        ))
        fig.add_hrect(y0=90, y1=110, fillcolor="green", opacity=0.1, line_width=0)
        fig.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Vibration (mm/s)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=machine_data['timestamp'],
            y=machine_data['vibration'],
            mode='lines',
            line=dict(color='#9b59b6', width=2)
        ))
        fig.add_hrect(y0=0.3, y1=0.6, fillcolor="green", opacity=0.1, line_width=0)
        fig.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Maintenance recommendations
    st.subheader("Maintenance Recommendations")
    
    recommendations = []
    latest = machine_data.iloc[-1]
    
    if latest['temperature'] > 85:
        recommendations.append({
            'severity': 'High',
            'component': 'Cooling System',
            'action': 'Check cooling system - temperature above normal',
            'estimated_days': np.random.randint(1, 5)
        })
    
    if latest['pressure'] > 120:
        recommendations.append({
            'severity': 'Medium',
            'component': 'Pressure Valve',
            'action': 'Inspect pressure valve - reading above threshold',
            'estimated_days': np.random.randint(5, 15)
        })
    
    if latest['vibration'] > 0.7:
        recommendations.append({
            'severity': 'High',
            'component': 'Bearings',
            'action': 'Check bearings - excessive vibration detected',
            'estimated_days': np.random.randint(1, 7)
        })
    
    if recommendations:
        for rec in recommendations:
            severity_color = "🔴" if rec['severity'] == 'High' else "🟡"
            st.markdown(
                f"""
                <div class="alert-box alert-{'critical' if rec['severity'] == 'High' else 'warning'}">
                    {severity_color} <strong>{rec['component']}</strong> - {rec['action']}
                    <br>Recommended action within: {rec['estimated_days']} days
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No immediate maintenance required")

def energy_optimization_tab(data):
    """Display energy optimization dashboard"""
    st.header("⚡ Energy Optimization")
    
    recent_data = data[data['timestamp'] > (datetime.now() - timedelta(days=7))]
    
    # Energy consumption trend
    st.subheader("Energy Consumption Trend")
    daily_energy = recent_data.groupby(recent_data['timestamp'].dt.date).agg({
        'power_consumption': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_energy['timestamp'],
        y=daily_energy['power_consumption'],
        marker_color='#f39c12',
        name='Daily Consumption'
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Energy Consumption (kWh)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Machine-wise energy consumption
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy by Machine")
        machine_energy = recent_data.groupby('machine_id')['power_consumption'].sum().reset_index()
        fig = px.pie(
            machine_energy,
            values='power_consumption',
            names='machine_id',
            title='Energy Distribution',
            color_discrete_sequence=px.colors.sequential.Oranges_r
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Energy Efficiency")
        machine_efficiency = recent_data.groupby('machine_id').agg({
            'power_consumption': 'mean',
            'production_count': 'mean'
        }).reset_index()
        machine_efficiency['efficiency'] = machine_efficiency['production_count'] / machine_efficiency['power_consumption']
        
        fig = px.bar(
            machine_efficiency,
            x='machine_id',
            y='efficiency',
            title='Units per kWh',
            color='efficiency',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Energy saving recommendations
    st.subheader("Energy Saving Recommendations")
    
    st.markdown("""
    <div class="alert-box alert-normal">
        💡 <strong>Recommendation 1:</strong> Schedule high-energy operations during off-peak hours to reduce costs by 15-20%
    </div>
    <div class="alert-box alert-normal">
        💡 <strong>Recommendation 2:</strong> Optimize machine idle time - potential savings of 500 kWh/day
    </div>
    <div class="alert-box alert-normal">
        💡 <strong>Recommendation 3:</strong> Implement variable speed drives on Machine M003 for 10-15% energy reduction
    </div>
    """, unsafe_allow_html=True)

def forecasting_tab(data):
    """Display demand forecasting dashboard"""
    st.header("📈 Demand Forecasting")
    
    st.markdown("""
        <div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
            <p style='margin: 0; font-size: 16px;'>
                💡 <strong>AI-Powered Forecasting:</strong> Using Facebook Prophet for accurate time series predictions
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for forecasting
    forecast_data = data.groupby(data['timestamp'].dt.date).agg({
        'production_count': 'sum'
    }).reset_index()
    forecast_data.columns = ['ds', 'y']
    
    # Forecast horizon selector
    col1, col2 = st.columns([3, 1])
    with col1:
        forecast_days = st.slider("📅 Forecast Horizon (days)", 1, 30, 7, help="Select number of days to forecast")
    with col2:
        st.metric("Data Points", len(forecast_data), help="Historical data points available")
    
    st.markdown("")  # Spacing
    
    if st.button("🚀 Generate Forecast", use_container_width=True, type="primary"):
        with st.spinner("🔄 Training Prophet model and generating forecast..."):
            try:
                # Train Prophet model
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False
                )
                model.fit(forecast_data)
                
                # Make future dataframe
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)
                
                st.success("✅ Forecast generated successfully!")
                st.markdown("")
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['y'],
                    mode='markers',
                    name='Historical Data',
                    marker=dict(color='#3498db', size=8, symbol='circle')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#e74c3c', width=3)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    fill='tonexty',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
                
                fig.update_layout(
                    title=f"{forecast_days}-Day Production Forecast",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Production Count",
                    hovermode='x unified',
                    font=dict(size=14)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast summary
                future_forecast = forecast[forecast['ds'] > forecast_data['ds'].max()]
                
                st.markdown("### 📊 Forecast Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Avg Daily Production",
                        f"{future_forecast['yhat'].mean():.0f}",
                        f"{((future_forecast['yhat'].mean() - forecast_data['y'].tail(7).mean()) / forecast_data['y'].tail(7).mean() * 100):.1f}%"
                    )
                with col2:
                    st.metric(
                        "Peak Production Day",
                        f"{future_forecast['yhat'].max():.0f}",
                        help="Maximum expected production"
                    )
                with col3:
                    st.metric(
                        "Low Production Day",
                        f"{future_forecast['yhat'].min():.0f}",
                        help="Minimum expected production"
                    )
                with col4:
                    st.metric(
                        "Total Forecast",
                        f"{future_forecast['yhat'].sum():.0f}",
                        f"{forecast_days} days"
                    )
                
                # Detailed forecast table
                st.markdown("### 📋 Detailed Forecast Data")
                forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_display.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
                forecast_display = forecast_display.round(0)
                st.dataframe(forecast_display, use_container_width=True, height=300)
                
            except Exception as e:
                st.error(f"❌ Error generating forecast: {str(e)}")
                st.info("💡 Tip: Ensure you have sufficient historical data for accurate forecasting")
    else:
        st.info("👆 Click **'Generate Forecast'** to see AI-powered production predictions")
        
        # Show preview of historical data
        st.markdown("### 📊 Historical Production Data Preview")
        recent_trend = forecast_data.tail(30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_trend['ds'],
            y=recent_trend['y'],
            mode='lines+markers',
            name='Production',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Last 30 Days Production Trend",
            height=350,
            xaxis_title="Date",
            yaxis_title="Production Count",
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

def alerts_tab(data):
    """Display alerts and notifications"""
    st.header("🚨 Alerts & Notifications")
    
    recent_data = data[data['timestamp'] > (datetime.now() - timedelta(hours=24))]
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.multiselect(
            "Severity",
            ["Critical", "Warning", "Info"],
            default=["Critical", "Warning"]
        )
    with col2:
        machine_filter = st.multiselect(
            "Machine",
            recent_data['machine_id'].unique(),
            default=recent_data['machine_id'].unique()
        )
    
    # Generate alerts
    alerts = []
    
    for idx, row in recent_data.iterrows():
        if row['machine_id'] not in machine_filter:
            continue
            
        # Critical alerts
        if row['temperature'] > 85:
            alerts.append({
                'severity': 'Critical',
                'machine': row['machine_id'],
                'message': f"High temperature detected: {row['temperature']:.1f}°F",
                'timestamp': row['timestamp'],
                'type': 'Temperature'
            })
        
        if row['vibration'] > 0.7:
            alerts.append({
                'severity': 'Critical',
                'machine': row['machine_id'],
                'message': f"Excessive vibration: {row['vibration']:.2f} mm/s",
                'timestamp': row['timestamp'],
                'type': 'Vibration'
            })
        
        # Warning alerts
        if 80 < row['temperature'] <= 85:
            alerts.append({
                'severity': 'Warning',
                'machine': row['machine_id'],
                'message': f"Elevated temperature: {row['temperature']:.1f}°F",
                'timestamp': row['timestamp'],
                'type': 'Temperature'
            })
        
        if row['quality_score'] < 90:
            alerts.append({
                'severity': 'Warning',
                'machine': row['machine_id'],
                'message': f"Low quality score: {row['quality_score']:.1f}%",
                'timestamp': row['timestamp'],
                'type': 'Quality'
            })
    
    # Filter by severity
    alerts_df = pd.DataFrame(alerts)
    if len(alerts_df) > 0:
        alerts_df = alerts_df[alerts_df['severity'].isin(severity_filter)]
        alerts_df = alerts_df.sort_values('timestamp', ascending=False)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            critical_count = len(alerts_df[alerts_df['severity'] == 'Critical'])
            st.metric("Critical Alerts", critical_count)
        with col2:
            warning_count = len(alerts_df[alerts_df['severity'] == 'Warning'])
            st.metric("Warning Alerts", warning_count)
        with col3:
            st.metric("Total Alerts", len(alerts_df))
        
        # Display alerts
        st.subheader("Recent Alerts")
        
        for idx, alert in alerts_df.head(20).iterrows():
            severity_class = "alert-critical" if alert['severity'] == 'Critical' else "alert-warning"
            severity_icon = "🔴" if alert['severity'] == 'Critical' else "🟡"
            
            st.markdown(
                f"""
                <div class="alert-box {severity_class}">
                    {severity_icon} <strong>[{alert['severity']}]</strong> {alert['machine']} - {alert['message']}
                    <br><small>Type: {alert['type']} | Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No alerts matching the selected filters")

def settings_tab():
    """Display settings and configuration"""
    st.header("⚙️ Settings & Configuration")
    
    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["🎚️ Alert Thresholds", "📧 Notifications", "💾 Data Management"])
    
    with tab1:
        st.subheader("Configure Alert Thresholds")
        st.markdown("Set custom thresholds for system alerts and notifications")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🌡️ Sensor Thresholds")
            temp = st.number_input("Temperature Threshold (°F)", value=85.0, step=0.5, help="Alert when temperature exceeds this value")
            pressure = st.number_input("Pressure Threshold (PSI)", value=120.0, step=1.0, help="Alert when pressure exceeds this value")
            vibration = st.number_input("Vibration Threshold (mm/s)", value=0.7, step=0.05, help="Alert when vibration exceeds this value")
        
        with col2:
            st.markdown("##### 📊 Production Thresholds")
            quality = st.number_input("Quality Score Threshold (%)", value=90.0, step=1.0, help="Alert when quality drops below this value")
            efficiency = st.number_input("Energy Efficiency Target", value=0.8, step=0.05, help="Target efficiency ratio")
            defect_rate = st.number_input("Defect Rate Threshold (%)", value=5.0, step=0.5, help="Alert when defect rate exceeds this value")
        
        st.markdown("")
        if st.button("💾 Save Threshold Settings", use_container_width=True, type="primary"):
            st.success("✅ Threshold settings saved successfully!")
            st.balloons()
    
    with tab2:
        st.subheader("Notification Preferences")
        st.markdown("Configure how you receive alerts and notifications")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📬 Notification Channels")
            email_notif = st.checkbox("📧 Email Notifications", value=True)
            sms_notif = st.checkbox("📱 SMS Notifications", value=False)
            dashboard_notif = st.checkbox("🖥️ Dashboard Alerts", value=True)
            push_notif = st.checkbox("🔔 Push Notifications", value=True)
        
        with col2:
            st.markdown("##### ⚙️ Notification Settings")
            notif_frequency = st.selectbox(
                "Alert Frequency",
                ["Immediate", "Every 15 minutes", "Every hour", "Daily digest"],
                help="How often to receive alert notifications"
            )
            severity_filter = st.multiselect(
                "Alert Severity Levels",
                ["Critical", "Warning", "Info"],
                default=["Critical", "Warning"],
                help="Which severity levels to receive"
            )
        
        st.markdown("")
        st.markdown("##### 📮 Contact Information")
        col1, col2 = st.columns(2)
        with col1:
            email = st.text_input("Email Address", value="admin@manusense.com", placeholder="your@email.com")
        with col2:
            phone = st.text_input("Phone Number", value="+1234567890", placeholder="+1234567890")
        
        st.markdown("")
        if st.button("💾 Save Notification Settings", use_container_width=True, type="primary"):
            st.success("✅ Notification settings saved successfully!")
    
    with tab3:
        st.subheader("Data Management")
        st.markdown("Manage your data, backups, and system cache")
        st.markdown("")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📤 Export Data")
            st.markdown("Download your manufacturing data")
            export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])
            if st.button("📥 Export Data", use_container_width=True):
                with st.spinner("Preparing export..."):
                    st.success("✅ Data exported successfully!")
                    st.info("📁 File saved to: exports/data_export.csv")
        
        with col2:
            st.markdown("##### 💾 Backup")
            st.markdown("Create system backup")
            backup_type = st.selectbox("Backup Type", ["Full", "Incremental", "Differential"])
            if st.button("🔄 Create Backup", use_container_width=True):
                with st.spinner("Creating backup..."):
                    st.success("✅ Backup completed successfully!")
                    st.info(f"📦 Backup saved with ID: BKP-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        with col3:
            st.markdown("##### 🧹 Maintenance")
            st.markdown("System maintenance tasks")
            st.markdown("")
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Cache cleared successfully!")
            
            if st.button("🔄 Reset Settings", use_container_width=True):
                st.warning("⚠️ This will reset all settings to default values")
        
        st.markdown("---")
        st.markdown("### 📊 Storage Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Storage", "500 GB")
        with col2:
            st.metric("Used", "342 GB", delta="68%")
        with col3:
            st.metric("Available", "158 GB")
        with col4:
            st.metric("Last Backup", "2 days ago")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6c757d; padding: 20px;'>
            <p style='font-size: 14px; margin: 0;'>ManuSense v1.0.0 | Last Updated: Sept 2026</p>
            <p style='font-size: 12px; margin-top: 5px;'>© 2026 ManuSense. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h1 style='color: #1f77b4; font-size: 36px; margin: 0;'>🏭 ManuSense</h1>
                <p style='color: #6c757d; font-size: 14px; margin-top: 5px;'>Smart Manufacturing Analytics</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
                <p style='color: white; margin: 0; font-size: 16px; font-weight: 600;'>
                    👤 {st.session_state.username}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "📍 Navigation",
            ["📊 Dashboard", "🎯 Quality Monitoring", "🔧 Predictive Maintenance", 
             "⚡ Energy Optimization", "📈 Forecasting", "🚨 Alerts", "⚙️ Settings"],
            label_visibility="visible"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Machines", "4", help="Active machines")
            st.metric("Operators", "3", help="Total operators")
        with col2:
            st.metric("Uptime", "98.5%", help="System uptime")
            st.metric("Alerts", "12", delta="-3", help="Active alerts")
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🔄 System Status")
        st.success("✅ All Systems Operational")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            logout()
    
    # Load data
    data = load_sample_data()
    
    # Main content
    if page == "📊 Dashboard":
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h1 style='font-size: 48px; margin-bottom: 10px;'>🏭 ManuSense Dashboard</h1>
                <p style='font-size: 20px; color: #6c757d;'>Real-time Manufacturing Analytics & Monitoring</p>
            </div>
        """, unsafe_allow_html=True)
        
        display_kpis(data)
        st.markdown("---")
        production_overview_tab(data)
        
    elif page == "🎯 Quality Monitoring":
        quality_monitoring_tab(data)
        
    elif page == "🔧 Predictive Maintenance":
        predictive_maintenance_tab(data)
        
    elif page == "⚡ Energy Optimization":
        energy_optimization_tab(data)
        
    elif page == "📈 Forecasting":
        forecasting_tab(data)
        
    elif page == "🚨 Alerts":
        alerts_tab(data)
        
    elif page == "⚙️ Settings":
        settings_tab()

if __name__ == "__main__":
    main()
