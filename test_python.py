import streamlit as st
import pandas as pd
import json
from datetime import datetime

st.set_page_config(
    page_title="Simple Patent Analyzer",
    page_icon="🔬",
    layout="wide"
)

def main():
    st.title("🔬 Simple Patent Intelligence System")
    st.markdown("### Basic Patent Data Analysis")
    
    # File upload with CSV/Excel support
    uploaded_file = st.file_uploader(
        "Upload patent data (CSV or Excel format)",
        type=['csv', 'xlsx', 'txt']
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                # Handle text files
                content = uploaded_file.read().decode('utf-8')
                st.text_area("File Content Preview", content[:1000])
                return
            
            st.success(f"✅ Loaded {len(df)} records")
            
            # Display data
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Basic statistics
            st.subheader("Basic Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Data Types", len(df.dtypes.unique()))
            
            # Column analysis
            st.subheader("Column Analysis")
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    st.write(f"**{col}**: {unique_count} unique values")
                    if unique_count < 20:
                        value_counts = df[col].value_counts().head(5)
                        st.bar_chart(value_counts)
                        
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Sample data demo
    if st.button("Load Sample Data"):
        sample_data = pd.DataFrame({
            'Patent_ID': ['US001', 'US002', 'US003', 'US004', 'US005'],
            'Title': ['AI System', 'ML Algorithm', 'Neural Network', 'Deep Learning', 'Computer Vision'],
            'Assignee': ['Company A', 'Company B', 'Company A', 'Company C', 'Company B'],
            'Year': [2021, 2022, 2022, 2023, 2023]
        })
        
        st.dataframe(sample_data)
        st.bar_chart(sample_data['Assignee'].value_counts())

if __name__ == "__main__":
    main()
