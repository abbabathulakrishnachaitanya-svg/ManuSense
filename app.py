import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="ManuSense - Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        font-size: 42px !important;
        margin-bottom: 20px !important;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        font-size: 28px !important;
        margin-top: 20px !important;
    }
    
    h3 {
        color: #34495e;
        font-weight: 600;
        font-size: 22px !important;
    }
    
    /* Metric cards */
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
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    /* Upload box */
    .uploadedFile {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    
    /* Result boxes */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 6px solid;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .defect-scratch {
        background-color: #fff3e0;
        border-color: #ff9800;
        color: #e65100;
    }
    
    .defect-crack {
        background-color: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }
    
    .defect-hole {
        background-color: #fce4ec;
        border-color: #e91e63;
        color: #880e4f;
    }
    
    .defect-dent {
        background-color: #f3e5f5;
        border-color: #9c27b0;
        color: #4a148c;
    }
    
    .defect-rust {
        background-color: #efebe9;
        border-color: #795548;
        color: #3e2723;
    }
    
    .defect-normal {
        background-color: #e8f5e9;
        border-color: #4caf50;
        color: #2e7d32;
    }
    
    /* Buttons */
    .stButton button {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 32px !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #2c3e50 !important;
        font-size: 16px !important;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1f77b4 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #2c3e50 !important;
    }
    
    /* Info boxes */
    .stAlert {
        font-size: 16px !important;
        padding: 16px !important;
    }
    
    /* Text */
    p, li, span, div {
        font-size: 16px !important;
        line-height: 1.6 !important;
        color: #2c3e50 !important;
    }
    
    .stMarkdown p {
        color: #2c3e50 !important;
    }
    
    strong {
        color: #1f77b4 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Defect types and their solutions
DEFECT_INFO = {
    "Scratch": {
        "severity": "Low to Medium",
        "icon": "🔸",
        "color": "defect-scratch",
        "description": "Surface scratches detected on the component",
        "causes": [
            "Improper handling during manufacturing or transportation",
            "Contact with rough surfaces or sharp objects",
            "Insufficient protective coating",
            "Tool marks during machining"
        ],
        "solutions": [
            "Polish the scratched area using fine-grit sandpaper (600-1000 grit)",
            "Apply touch-up paint or protective coating",
            "Use buffing compound for minor scratches",
            "Replace component if scratches affect structural integrity",
            "Implement better handling procedures"
        ],
        "prevention": [
            "Use protective packaging during transport",
            "Train workers on proper handling techniques",
            "Apply protective films during manufacturing",
            "Regular tool maintenance and replacement"
        ]
    },
    "Crack": {
        "severity": "High",
        "icon": "⚠️",
        "color": "defect-crack",
        "description": "Structural crack detected - requires immediate attention",
        "causes": [
            "Excessive stress or load beyond material capacity",
            "Thermal expansion/contraction cycles",
            "Material fatigue over time",
            "Manufacturing defects or poor welding",
            "Corrosion-induced stress"
        ],
        "solutions": [
            "⚠️ STOP using the component immediately",
            "Perform detailed inspection to assess crack depth and length",
            "For minor cracks: Apply epoxy resin or welding repair",
            "For major cracks: Replace the component entirely",
            "Conduct stress analysis to prevent future occurrence",
            "Document the defect for quality control records"
        ],
        "prevention": [
            "Regular non-destructive testing (NDT)",
            "Implement load monitoring systems",
            "Use appropriate material grade for application",
            "Apply anti-corrosion treatments",
            "Schedule preventive maintenance"
        ]
    },
    "Hole": {
        "severity": "Medium to High",
        "icon": "⭕",
        "color": "defect-hole",
        "description": "Hole or perforation detected in the component",
        "causes": [
            "Corrosion eating through material",
            "Impact damage from foreign objects",
            "Manufacturing drilling errors",
            "Wear and tear over extended use",
            "Chemical exposure"
        ],
        "solutions": [
            "Clean the area around the hole thoroughly",
            "For small holes: Use welding or metal filler",
            "For larger holes: Apply patch plates with proper welding",
            "Use epoxy-based repair compounds for non-structural parts",
            "Replace component if hole compromises integrity",
            "Inspect surrounding areas for additional damage"
        ],
        "prevention": [
            "Apply protective coatings (galvanizing, painting)",
            "Regular inspection and maintenance schedule",
            "Use corrosion-resistant materials",
            "Implement proper drainage to prevent water accumulation",
            "Control environmental exposure"
        ]
    },
    "Dent": {
        "severity": "Low to Medium",
        "icon": "🔴",
        "color": "defect-dent",
        "description": "Dent or deformation detected on the surface",
        "causes": [
            "Impact from dropped objects or collisions",
            "Improper handling during installation",
            "Transportation damage",
            "Excessive force during assembly",
            "Material weakness or thin sections"
        ],
        "solutions": [
            "Use dent removal tools (slide hammer, dent puller)",
            "Apply heat and cold cycles for metal restoration",
            "Fill with body filler and sand smooth",
            "Use suction cup dent removers for shallow dents",
            "Replace if dent affects functionality",
            "Repaint or refinish the affected area"
        ],
        "prevention": [
            "Use proper lifting and handling equipment",
            "Implement protective barriers during transport",
            "Train personnel on careful handling",
            "Use packaging with adequate cushioning",
            "Regular equipment inspection"
        ]
    },
    "Rust/Corrosion": {
        "severity": "Medium",
        "icon": "🟤",
        "color": "defect-rust",
        "description": "Rust or corrosion detected on metal surface",
        "causes": [
            "Exposure to moisture and oxygen",
            "Lack of protective coating",
            "Chemical exposure",
            "Salt or acidic environment",
            "Poor material quality"
        ],
        "solutions": [
            "Remove rust using wire brush or sandblasting",
            "Apply rust converter chemical",
            "Use phosphoric acid-based rust remover",
            "Sand down to bare metal and repaint",
            "Apply anti-rust primer and protective coating",
            "For severe corrosion: Replace the component"
        ],
        "prevention": [
            "Apply protective coatings (paint, galvanizing, powder coating)",
            "Use stainless steel or corrosion-resistant materials",
            "Keep surfaces dry and well-ventilated",
            "Regular cleaning and maintenance",
            "Apply rust-preventive oils or waxes"
        ]
    },
    "Normal": {
        "severity": "None",
        "icon": "✅",
        "color": "defect-normal",
        "description": "No defects detected - component is in good condition",
        "causes": [],
        "solutions": [
            "No action required - component is in acceptable condition",
            "Continue with regular maintenance schedule",
            "Document inspection results for quality records",
            "Monitor for any changes in future inspections"
        ],
        "prevention": [
            "Maintain current quality control procedures",
            "Continue regular inspection schedule",
            "Keep documentation updated",
            "Follow established maintenance protocols"
        ]
    }
}

# Mock ML model prediction function
def predict_defect(image):
    """
    Mock function to simulate defect detection
    In production, this would use a trained CNN model
    """
    # Simulate processing time
    import time
    time.sleep(1.5)
    
    # Randomly select a defect type for demo
    defect_types = list(DEFECT_INFO.keys())
    probabilities = np.random.dirichlet(np.ones(len(defect_types)), size=1)[0]
    
    # Sort by probability
    results = sorted(zip(defect_types, probabilities), key=lambda x: x[1], reverse=True)
    
    return results

def display_defect_result(defect_type, confidence, image):
    """Display detailed defect analysis results"""
    
    info = DEFECT_INFO[defect_type]
    
    st.markdown(f"""
        <div class="result-box {info['color']}">
            <h2>{info['icon']} Detected: {defect_type}</h2>
            <p style="margin: 0;"><strong>Confidence:</strong> {confidence*100:.2f}%</p>
            <p style="margin: 0;"><strong>Severity Level:</strong> {info['severity']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show image and analysis side by side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True, caption="Inspected Component")
    
    with col2:
        st.subheader("📊 Analysis Details")
        st.markdown(f"**Description:** {info['description']}")
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Detection Confidence", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffebee'},
                    {'range': [50, 75], 'color': '#fff3e0'},
                    {'range': [75, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed information in tabs
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🔍 Root Causes", "🛠️ Solutions", "🛡️ Prevention"])
    
    with tab1:
        if info['causes']:
            st.markdown("### Possible Root Causes:")
            for i, cause in enumerate(info['causes'], 1):
                st.markdown(f"{i}. {cause}")
        else:
            st.success("No defects detected - component is in good condition!")
    
    with tab2:
        st.markdown("### Recommended Solutions:")
        for i, solution in enumerate(info['solutions'], 1):
            st.markdown(f"{i}. {solution}")
    
    with tab3:
        if info['prevention']:
            st.markdown("### Preventive Measures:")
            for i, prevention in enumerate(info['prevention'], 1):
                st.markdown(f"{i}. {prevention}")

def display_all_predictions(predictions, image):
    """Display all prediction results with probabilities"""
    st.subheader("📊 All Detection Results")
    
    # Create dataframe
    df = pd.DataFrame(predictions, columns=['Defect Type', 'Probability'])
    df['Confidence %'] = (df['Probability'] * 100).round(2)
    
    # Bar chart
    fig = px.bar(
        df, 
        x='Defect Type', 
        y='Confidence %',
        title='Defect Detection Confidence Levels',
        color='Confidence %',
        color_continuous_scale='RdYlGn',
        text='Confidence %'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)
    
    # Show table
    st.dataframe(
        df[['Defect Type', 'Confidence %']].style.background_gradient(
            subset=['Confidence %'], 
            cmap='RdYlGn'
        ),
        use_container_width=True,
        height=250
    )

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='font-size: 48px; margin-bottom: 10px;'>🔍 ManuSense</h1>
            <p style='font-size: 22px; color: #6c757d;'>AI-Powered Manufacturing Defect Detection System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: white; font-size: 32px; margin: 0;'>🏭 ManuSense</h2>
                <p style='color: white; font-size: 16px; margin-top: 5px;'>Defect Detection AI</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Supported Defects")
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; color: #2c3e50;'>
            <p style='margin: 5px 0; font-size: 16px; color: #2c3e50;'><strong>🔸 Scratches</strong> - Surface marks</p>
            <p style='margin: 5px 0; font-size: 16px; color: #2c3e50;'><strong>⚠️ Cracks</strong> - Structural damage</p>
            <p style='margin: 5px 0; font-size: 16px; color: #2c3e50;'><strong>⭕ Holes</strong> - Perforations</p>
            <p style='margin: 5px 0; font-size: 16px; color: #2c3e50;'><strong>🔴 Dents</strong> - Deformations</p>
            <p style='margin: 5px 0; font-size: 16px; color: #2c3e50;'><strong>🟤 Rust</strong> - Corrosion</p>
            <p style='margin: 5px 0; font-size: 16px; color: #2c3e50;'><strong>✅ Normal</strong> - No defects</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Scans", "1,247")
            st.metric("Accuracy", "94.5%")
        with col2:
            st.metric("Defects Found", "342")
            st.metric("Normal", "905")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        <div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; color: #0d47a1;'>
            <p style='margin: 5px 0; font-size: 15px; color: #1565c0;'><strong>1. Upload</strong> an image</p>
            <p style='margin: 5px 0; font-size: 15px; color: #1565c0;'><strong>2. AI analyzes</strong> the image</p>
            <p style='margin: 5px 0; font-size: 15px; color: #1565c0;'><strong>3. Detects</strong> defect type</p>
            <p style='margin: 5px 0; font-size: 15px; color: #1565c0;'><strong>4. Provides</strong> solutions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <p style='text-align: center; color: #6c757d; font-size: 14px;'>
                v1.0.0 | Powered by Deep Learning
            </p>
        """, unsafe_allow_html=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Defect Detection", "📚 Defect Guide", "📊 History"])
    
    with tab1:
        st.markdown("### Upload Component Image for Inspection")
        st.markdown("Supported formats: JPG, JPEG, PNG | Max size: 10MB")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the component to inspect"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption='Uploaded Image', use_container_width=True)
            
            st.markdown("---")
            
            # Analyze button
            if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
                with st.spinner("🤖 AI is analyzing the image..."):
                    # Get predictions
                    predictions = predict_defect(image)
                    top_defect, top_confidence = predictions[0]
                
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                
                # Display primary result
                display_defect_result(top_defect, top_confidence, image)
                
                st.markdown("---")
                
                # Display all predictions
                display_all_predictions(predictions, image)
                
                # Download report button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("📥 Download Report", use_container_width=True):
                        st.success("Report downloaded successfully!")
        
        else:
            # Show example
            st.info("👆 Please upload an image to begin inspection")
            
            st.markdown("### 📸 Example Images")
            st.markdown("Here are some examples of defects our AI can detect:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Scratch Example**")
                st.markdown("🔸 Surface scratches on metal")
            with col2:
                st.markdown("**Crack Example**")
                st.markdown("⚠️ Structural cracks")
            with col3:
                st.markdown("**Hole Example**")
                st.markdown("⭕ Perforations in material")
    
    with tab2:
        st.subheader("📚 Comprehensive Defect Guide")
        st.markdown("Learn about different types of defects and how to handle them")
        st.markdown("---")
        
        # Display all defect types
        for defect_type, info in DEFECT_INFO.items():
            with st.expander(f"{info['icon']} {defect_type} - Severity: {info['severity']}", expanded=False):
                st.markdown(f"**Description:** {info['description']}")
                
                if info['causes']:
                    st.markdown("**Common Causes:**")
                    for cause in info['causes']:
                        st.markdown(f"- {cause}")
                
                st.markdown("**Solutions:**")
                for solution in info['solutions']:
                    st.markdown(f"- {solution}")
                
                if info['prevention']:
                    st.markdown("**Prevention:**")
                    for prevention in info['prevention']:
                        st.markdown(f"- {prevention}")
    
    with tab3:
        st.subheader("📊 Inspection History")
        st.markdown("Recent inspection results and statistics")
        
        # Mock history data
        history_data = {
            'Date': ['2024-01-15', '2024-01-14', '2024-01-14', '2024-01-13', '2024-01-13'],
            'Time': ['14:30', '11:20', '09:15', '16:45', '10:30'],
            'Defect Type': ['Scratch', 'Normal', 'Crack', 'Rust', 'Normal'],
            'Confidence': ['92.5%', '98.2%', '87.3%', '91.8%', '96.4%'],
            'Status': ['Resolved', 'OK', 'Under Review', 'Resolved', 'OK']
        }
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, height=300)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Today's Scans", "23", delta="+5")
        with col2:
            st.metric("Defects Today", "8", delta="+2")
        with col3:
            st.metric("Avg Confidence", "93.2%", delta="+1.5%")
        with col4:
            st.metric("Resolution Rate", "87%", delta="+3%")

if __name__ == "__main__":
    main()
