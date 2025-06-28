# test_python.py
import streamlit as st
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Using basic charts.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

import re
from datetime import datetime
import numpy as np
from collections import Counter

st.set_page_config(
    page_title="Patent Intelligence System",
    page_icon="🔬",
    layout="wide"
)

def safe_process_docx(uploaded_file):
    """Safely process DOCX file with fallback"""
    if not DOCX_AVAILABLE:
        st.error("python-docx not available. Please check requirements.txt")
        return pd.DataFrame()
    
    try:
        doc = docx.Document(uploaded_file)
        patents_data = []
        current_patent = {}
        
        for paragraph in doc.paragrapars:
            text = paragraph.text.strip()
            if not text:
                continue
                
            # Simple text parsing
            if "patent" in text.lower() and len(current_patent) > 0:
                patents_data.append(current_patent)
                current_patent = {}
            
            # Extract basic info
            if ":" in text:
                key, value = text.split(":", 1)
                key = key.lower().strip()
                value = value.strip()
                
                if "patent" in key or "number" in key:
                    current_patent['patent_number'] = value
                elif "title" in key:
                    current_patent['title'] = value
                elif "inventor" in key:
                    current_patent['inventors'] = value
                elif "assignee" in key:
                    current_patent['assignee'] = value
                elif "date" in key:
                    current_patent['date'] = value
                elif "abstract" in key:
                    current_patent['abstract'] = value
            elif len(text) > 50 and 'abstract' not in current_patent:
                current_patent['abstract'] = text
                
        if current_patent:
            patents_data.append(current_patent)
            
        return pd.DataFrame(patents_data)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return pd.DataFrame()

def create_simple_chart(data, chart_type="bar"):
    """Create charts with fallback to basic Streamlit charts"""
    if PLOTLY_AVAILABLE:
        if chart_type == "bar":
            fig = px.bar(x=data.index, y=data.values, title="Analysis Results")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "line":
            fig = px.line(x=data.index, y=data.values, title="Trend Analysis")
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to Streamlit's built-in charts
        if chart_type == "bar":
            st.bar_chart(data)
        elif chart_type == "line":
            st.line_chart(data)

def main():
    st.title("🔬 Patent Intelligence Analysis System")
    st.markdown("### Upload and Analyze Patent Documents")
    
    # Check dependencies
    st.sidebar.title("System Status")
    st.sidebar.write("✅ Streamlit: Available")
    st.sidebar.write("✅ Pandas: Available")
    st.sidebar.write(f"{'✅' if PLOTLY_AVAILABLE else '❌'} Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
    st.sidebar.write(f"{'✅' if SKLEARN_AVAILABLE else '❌'} Scikit-learn: {'Available' if SKLEARN_AVAILABLE else 'Not Available'}")
    st.sidebar.write(f"{'✅' if DOCX_AVAILABLE else '❌'} Python-docx: {'Available' if DOCX_AVAILABLE else 'Not Available'}")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Analysis", 
                               ["Upload Data", "Basic Analysis", "Sample Demo"])
    
    if page == "Upload Data":
        st.header("📤 Upload Patent Data")
        
        if not DOCX_AVAILABLE:
            st.error("⚠️ python-docx package not available. Please check your requirements.txt file.")
            st.code("""
# Add this to your requirements.txt:
python-docx==0.8.11
            """)
            return
        
        uploaded_file = st.file_uploader(
            "Choose a DOCX file with patent information",
            type=['docx']
        )
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                df = safe_process_docx(uploaded_file)
                
            if not df.empty:
                st.success(f"✅ Processed {len(df)} patents")
                st.session_state['patent_data'] = df
                st.dataframe(df)
            else:
                st.error("No data could be extracted from the document")
    
    elif page == "Basic Analysis":
        if 'patent_data' not in st.session_state:
            st.warning("Please upload data first")
            return
            
        st.header("📊 Basic Patent Analysis")
        df = st.session_state['patent_data']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patents", len(df))
        with col2:
            if 'assignee' in df.columns:
                st.metric("Unique Assignees", df['assignee'].nunique())
        with col3:
            if 'inventors' in df.columns:
                inventor_count = len([inv for inventors in df['inventors'].dropna() 
                                    for inv in str(inventors).split(',')])
                st.metric("Total Inventors", inventor_count)
        
        # Simple visualizations
        if 'assignee' in df.columns:
            st.subheader("Top Assignees")
            assignee_counts = df['assignee'].value_counts().head(10)
            create_simple_chart(assignee_counts, "bar")
    
    elif page == "Sample Demo":
        st.header("🎯 Sample Patent Analysis Demo")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'patent_number': ['US10123456', 'US10123457', 'US10123458', 'US10123459', 'US10123460'],
            'title': ['AI System for Data Analysis', 'Machine Learning Algorithm', 'Neural Network Architecture', 'Deep Learning Model', 'Computer Vision System'],
            'assignee': ['TechCorp Inc', 'AI Innovations', 'TechCorp Inc', 'DataSoft LLC', 'AI Innovations'],
            'year': [2021, 2022, 2022, 2023, 2023]
        })
        
        st.subheader("Sample Patent Dataset")
        st.dataframe(sample_data)
        
        # Sample analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patents by Assignee")
            assignee_counts = sample_data['assignee'].value_counts()
            create_simple_chart(assignee_counts, "bar")
        
        with col2:
            st.subheader("Patents by Year")
            year_counts = sample_data['year'].value_counts().sort_index()
            create_simple_chart(year_counts, "line")

if __name__ == "__main__":
    main()
