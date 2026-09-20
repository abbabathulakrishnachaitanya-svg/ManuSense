import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration - LIGHT THEME
st.set_page_config(
    page_title="ManuSense - Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme and improve text visibility
st.markdown("""
    <style>
    /* Force light background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main content area */
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    
    /* ALL TEXT - Make it dark and visible */
    * {
        color: #1a1a1a !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 48px !important;
    }
    
    h2 {
        font-size: 32px !important;
    }
    
    h3 {
        font-size: 24px !important;
    }
    
    /* Paragraphs and text */
    p, span, div, li, label {
        color: #1a1a1a !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    [data-testid="stMetric"] label {
        font-size: 18px !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1f77b4 !important;
        color: white !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        padding: 15px 40px !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background-color: #1557a0 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #f8f9fa;
        border: 3px dashed #1f77b4;
        border-radius: 15px;
        padding: 30px;
    }
    
    [data-testid="stFileUploader"] label {
        font-size: 20px !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        color: #1a1a1a !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 15px 30px !important;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background-color: #e3f2fd !important;
        border-left: 5px solid #1f77b4 !important;
        padding: 20px !important;
        font-size: 18px !important;
    }
    
    .stAlert * {
        color: #0d47a1 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }
    
    /* Dataframe */
    .dataframe {
        font-size: 16px !important;
        color: #1a1a1a !important;
    }
    
    /* Make sure markdown text is visible */
    .stMarkdown {
        color: #1a1a1a !important;
    }
    
    .stMarkdown * {
        color: #1a1a1a !important;
    }
    </style>
""", unsafe_allow_html=True)

# Defect information database
DEFECT_INFO = {
    "Scratch": {
        "emoji": "🔸",
        "severity": "Low to Medium",
        "color": "#ff9800",
        "description": "Surface scratches or abrasions detected on the component surface.",
        "causes": [
            "Improper handling during manufacturing or transportation",
            "Contact with rough surfaces or sharp objects",
            "Insufficient protective coating during processing",
            "Tool marks during machining operations"
        ],
        "solutions": [
            "Polish the scratched area using fine-grit sandpaper (600-1000 grit)",
            "Apply touch-up paint or protective coating to prevent rust",
            "Use buffing compound for minor surface scratches",
            "Replace component if scratches affect structural integrity",
            "Implement better handling procedures to prevent future scratches"
        ],
        "prevention": [
            "Use protective packaging during transportation",
            "Train workers on proper handling techniques",
            "Apply protective films during manufacturing process",
            "Regular tool maintenance and replacement schedule"
        ]
    },
    "Crack": {
        "emoji": "⚠️",
        "severity": "HIGH - Critical",
        "color": "#f44336",
        "description": "Structural crack detected - IMMEDIATE ACTION REQUIRED!",
        "causes": [
            "Excessive stress or load beyond material capacity",
            "Thermal expansion and contraction cycles",
            "Material fatigue from repeated loading",
            "Poor welding quality or manufacturing defects",
            "Corrosion-induced stress cracking"
        ],
        "solutions": [
            "⚠️ STOP USING THE COMPONENT IMMEDIATELY",
            "Perform detailed crack inspection (depth, length, location)",
            "For minor surface cracks: Grind out and weld repair",
            "For through-cracks: REPLACE COMPONENT - DO NOT REPAIR",
            "Conduct root cause analysis and stress testing",
            "Document defect for quality control and safety records"
        ],
        "prevention": [
            "Regular non-destructive testing (NDT) - ultrasonic, X-ray",
            "Implement load monitoring and tracking systems",
            "Use appropriate material grade for the application",
            "Apply anti-corrosion treatments regularly",
            "Schedule preventive maintenance and inspections"
        ]
    },
    "Hole": {
        "emoji": "⭕",
        "severity": "Medium to High",
        "color": "#e91e63",
        "description": "Hole or perforation detected in the component structure.",
        "causes": [
            "Corrosion eating through the material thickness",
            "Impact damage from foreign objects or tools",
            "Manufacturing drilling errors or misalignment",
            "Long-term wear and material degradation",
            "Chemical exposure causing material breakdown"
        ],
        "solutions": [
            "Clean and prepare the area around the hole",
            "For small holes (< 5mm): Use welding or metal filler",
            "For larger holes: Apply patch plate with proper welding",
            "Use epoxy-based repair compounds for non-structural parts",
            "Replace entire component if hole affects critical function",
            "Inspect surrounding areas for additional damage"
        ],
        "prevention": [
            "Apply protective coatings (galvanizing, painting, powder coating)",
            "Establish regular inspection and maintenance schedule",
            "Use corrosion-resistant materials for harsh environments",
            "Implement proper drainage to prevent water accumulation",
            "Control environmental exposure and chemical contact"
        ]
    },
    "Dent": {
        "emoji": "🔴",
        "severity": "Low to Medium",
        "color": "#9c27b0",
        "description": "Dent or surface deformation detected on the component.",
        "causes": [
            "Impact from dropped objects or collisions during handling",
            "Improper installation or assembly procedures",
            "Transportation damage from inadequate protection",
            "Excessive force applied during manufacturing",
            "Material weakness in thin sections"
        ],
        "solutions": [
            "Use dent removal tools (slide hammer, dent puller)",
            "Apply heat and cold cycles to restore metal shape",
            "Fill deeper dents with body filler and sand smooth",
            "Use suction-based dent removers for shallow depressions",
            "Replace component if dent affects functionality",
            "Refinish or repaint the affected surface area"
        ],
        "prevention": [
            "Use proper lifting and handling equipment always",
            "Implement protective barriers during transportation",
            "Train all personnel on careful handling procedures",
            "Use adequate cushioning in packaging materials",
            "Conduct regular equipment condition inspections"
        ]
    },
    "Rust": {
        "emoji": "🟤",
        "severity": "Medium",
        "color": "#795548",
        "description": "Rust or corrosion detected on metal surfaces.",
        "causes": [
            "Prolonged exposure to moisture and oxygen",
            "Lack of protective coating or paint damage",
            "Chemical exposure (salt, acids, industrial fumes)",
            "High humidity or wet environment conditions",
            "Poor quality material or inadequate surface treatment"
        ],
        "solutions": [
            "Remove rust using wire brush, grinder, or sandblasting",
            "Apply rust converter chemical to neutralize corrosion",
            "Use phosphoric acid-based rust remover products",
            "Sand down to bare metal for complete rust removal",
            "Apply anti-rust primer followed by protective paint",
            "For severe corrosion: Replace the affected component"
        ],
        "prevention": [
            "Apply protective coatings (paint, galvanizing, powder coat)",
            "Use stainless steel or corrosion-resistant alloys",
            "Keep surfaces dry and ensure proper ventilation",
            "Regular cleaning and maintenance schedule",
            "Apply rust-preventive oils, waxes, or inhibitors"
        ]
    },
    "Normal": {
        "emoji": "✅",
        "severity": "None - Good Condition",
        "color": "#4caf50",
        "description": "No defects detected - component is in acceptable condition.",
        "causes": [],
        "solutions": [
            "✅ No immediate action required",
            "Continue with standard maintenance schedule",
            "Document inspection results for quality records",
            "Monitor component in future inspections",
            "Maintain current operational procedures"
        ],
        "prevention": [
            "Maintain current quality control procedures",
            "Continue regular inspection schedule",
            "Keep accurate documentation and records",
            "Follow established maintenance protocols",
            "Train staff on proper handling and care"
        ]
    }
}

def predict_defect(image):
    """Simulate AI defect detection"""
    import time
    time.sleep(2)  # Simulate processing
    
    # Generate random predictions for demo
    defect_types = list(DEFECT_INFO.keys())
    probabilities = np.random.dirichlet(np.ones(len(defect_types)), size=1)[0]
    
    # Sort by confidence
    results = sorted(zip(defect_types, probabilities), key=lambda x: x[1], reverse=True)
    return results

def main():
    # Title
    st.markdown("""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;'>
            <h1 style='color: white !important; font-size: 56px; margin: 0;'>🔍 ManuSense</h1>
            <p style='color: white !important; font-size: 26px; margin-top: 10px; font-weight: 600;'>AI-Powered Defect Detection System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🏭 ManuSense")
        st.markdown("### Defect Detection AI")
        st.markdown("---")
        
        st.markdown("### 📋 Defects We Detect:")
        st.markdown("**🔸 Scratches** - Surface marks")
        st.markdown("**⚠️ Cracks** - Critical damage")
        st.markdown("**⭕ Holes** - Perforations")  
        st.markdown("**🔴 Dents** - Deformations")
        st.markdown("**🟤 Rust** - Corrosion")
        st.markdown("**✅ Normal** - No defects")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        st.metric("Total Scans", "1,247")
        st.metric("Defects Found", "342")
        st.metric("Accuracy", "94.5%")
        
        st.markdown("---")
        st.markdown("### 💡 How It Works")
        st.markdown("**1.** Upload image")
        st.markdown("**2.** AI analyzes it")
        st.markdown("**3.** Get defect type")
        st.markdown("**4.** View solutions")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Detect Defects", "📚 Defect Guide", "📊 History"])
    
    with tab1:
        st.markdown("## Upload Component Image")
        st.markdown("**Supported:** JPG, JPEG, PNG | **Max Size:** 10MB")
        st.markdown("")
        
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            st.markdown("---")
            
            if st.button("🚀 ANALYZE IMAGE NOW", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your image..."):
                    predictions = predict_defect(image)
                    top_defect, top_confidence = predictions[0]
                
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                
                # Show result
                info = DEFECT_INFO[top_defect]
                
                st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 30px; border-radius: 15px; border-left: 8px solid {info['color']}; margin: 20px 0;'>
                        <h1 style='color: {info['color']} !important; margin: 0;'>{info['emoji']} {top_defect} Detected</h1>
                        <h3 style='color: #1a1a1a !important; margin-top: 10px;'>Confidence: {top_confidence*100:.1f}%</h3>
                        <h3 style='color: #1a1a1a !important;'>Severity: {info['severity']}</h3>
                        <p style='color: #1a1a1a !important; font-size: 20px; margin-top: 15px;'>{info['description']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Solutions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔍 Root Causes:")
                    if info['causes']:
                        for i, cause in enumerate(info['causes'], 1):
                            st.markdown(f"**{i}.** {cause}")
                    
                    st.markdown("### 🛡️ Prevention:")
                    if info['prevention']:
                        for i, prev in enumerate(info['prevention'], 1):
                            st.markdown(f"**{i}.** {prev}")
                
                with col2:
                    st.markdown("### 🛠️ Solutions:")
                    for i, solution in enumerate(info['solutions'], 1):
                        st.markdown(f"**{i}.** {solution}")
                
                # All predictions
                st.markdown("---")
                st.markdown("## 📊 All Detection Results")
                
                df = pd.DataFrame(predictions, columns=['Defect Type', 'Probability'])
                df['Confidence %'] = (df['Probability'] * 100).round(1)
                
                fig = px.bar(df, x='Defect Type', y='Confidence %', 
                           title='Confidence Levels for All Defect Types',
                           color='Confidence %', color_continuous_scale='RdYlGn')
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                fig.update_layout(height=400, font=dict(size=16))
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Please upload an image to start defect detection")
    
    with tab2:
        st.markdown("## 📚 Complete Defect Reference Guide")
        st.markdown("Learn about all defect types and how to handle them")
        st.markdown("---")
        
        for defect_type, info in DEFECT_INFO.items():
            with st.expander(f"{info['emoji']} {defect_type} - Severity: {info['severity']}", expanded=False):
                st.markdown(f"**Description:** {info['description']}")
                
                if info['causes']:
                    st.markdown("**Common Causes:**")
                    for cause in info['causes']:
                        st.markdown(f"• {cause}")
                
                st.markdown("**Recommended Solutions:**")
                for solution in info['solutions']:
                    st.markdown(f"• {solution}")
                
                if info['prevention']:
                    st.markdown("**Prevention Measures:**")
                    for prevention in info['prevention']:
                        st.markdown(f"• {prevention}")
    
    with tab3:
        st.markdown("## 📊 Inspection History")
        
        history_data = {
            'Date': ['2024-01-15', '2024-01-14', '2024-01-14', '2024-01-13'],
            'Time': ['14:30', '11:20', '09:15', '16:45'],
            'Defect': ['Scratch', 'Normal', 'Crack', 'Rust'],
            'Confidence': ['92.5%', '98.2%', '87.3%', '91.8%'],
            'Status': ['Resolved', 'OK', 'Under Review', 'Resolved']
        }
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, height=300)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Today's Scans", "23", delta="+5")
        with col2:
            st.metric("Defects Found", "8", delta="+2")
        with col3:
            st.metric("Avg Confidence", "93.2%")
        with col4:
            st.metric("Resolution Rate", "87%")

if __name__ == "__main__":
    main()
