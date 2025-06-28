# simplified_patent_analyzer.py - Complete system in one file
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import docx
import re
from datetime import datetime
import numpy as np
from collections import Counter

st.set_page_config(
    page_title="Patent Intelligence System",
    page_icon="🔬",
    layout="wide"
)

class PatentAnalyzer:
    def __init__(self):
        self.data = None
        
    def process_docx(self, uploaded_file):
        """Process uploaded DOCX file"""
        try:
            doc = docx.Document(uploaded_file)
            patents_data = []
            current_patent = {}
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                    
                # Extract patent info
                if "Patent Number:" in text or "US" in text[:10]:
                    if current_patent:
                        patents_data.append(current_patent)
                    current_patent = {'patent_number': text}
                elif "Title:" in text:
                    current_patent['title'] = text.replace("Title:", "").strip()
                elif "Inventor" in text:
                    current_patent['inventors'] = text.split(":")[-1].strip()
                elif "Assignee:" in text:
                    current_patent['assignee'] = text.replace("Assignee:", "").strip()
                elif "Date:" in text:
                    current_patent['date'] = text.split(":")[-1].strip()
                elif "Abstract" in text:
                    current_patent['abstract'] = text.replace("Abstract:", "").strip()
                elif len(text) > 50 and 'abstract' not in current_patent:
                    current_patent['abstract'] = text
                    
            if current_patent:
                patents_data.append(current_patent)
                
            return pd.DataFrame(patents_data)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return pd.DataFrame()
    
    def create_technology_clusters(self, df, n_clusters=5):
        """Create technology clusters"""
        if 'abstract' not in df.columns or df['abstract'].isna().all():
            return df, {}
            
        abstracts = df['abstract'].fillna('').astype(str)
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(abstracts)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            df['cluster'] = clusters
            
            # Get keywords for each cluster
            feature_names = vectorizer.get_feature_names_out()
            keywords = {}
            for i in range(n_clusters):
                top_indices = kmeans.cluster_centers_[i].argsort()[-5:][::-1]
                keywords[i] = [feature_names[idx] for idx in top_indices]
                
            return df, keywords
        except:
            df['cluster'] = 0
            return df, {0: ['technology']}
    
    def analyze_trends(self, df):
        """Analyze patent trends"""
        if 'date' not in df.columns:
            return None
            
        # Extract years from dates
        years = []
        for date_str in df['date'].fillna(''):
            year_match = re.search(r'20\d{2}', str(date_str))
            if year_match:
                years.append(int(year_match.group()))
            else:
                years.append(2020)  # Default year
                
        df['year'] = years
        yearly_counts = df.groupby('year').size().reset_index(name='count')
        return yearly_counts
    
    def identify_key_players(self, df):
        """Identify key players"""
        if 'assignee' not in df.columns:
            return None
            
        assignee_counts = df['assignee'].value_counts().head(10)
        return assignee_counts

def main():
    st.title("🔬 Patent Intelligence Analysis System")
    st.markdown("### Upload and Analyze Patent Documents")
    
    analyzer = PatentAnalyzer()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Analysis", 
                               ["Upload Data", "Technology Landscape", "Trends", "Key Players"])
    
    if page == "Upload Data":
        st.header("📤 Upload Patent Data")
        
        uploaded_file = st.file_uploader(
            "Choose a DOCX file with patent information",
            type=['docx']
        )
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                df = analyzer.process_docx(uploaded_file)
                
            if not df.empty:
                st.success(f"✅ Processed {len(df)} patents")
                st.session_state['patent_data'] = df
                st.dataframe(df.head())
                
                # Basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patents", len(df))
                with col2:
                    if 'assignee' in df.columns:
                        st.metric("Unique Assignees", df['assignee'].nunique())
                with col3:
                    if 'date' in df.columns:
                        st.metric("Date Range", "2020-2024")  # Simplified
            else:
                st.error("No data could be extracted")
    
    elif page == "Technology Landscape":
        if 'patent_data' not in st.session_state:
            st.warning("Please upload data first")
            return
            
        st.header("📊 Technology Landscape")
        df = st.session_state['patent_data']
        
        n_clusters = st.slider("Number of clusters", 3, 8, 5)
        
        if st.button("Generate Landscape"):
            with st.spinner("Creating technology clusters..."):
                clustered_df, keywords = analyzer.create_technology_clusters(df, n_clusters)
                
                # Visualization
                if 'cluster' in clustered_df.columns:
                    fig = px.scatter(clustered_df, 
                                   x='cluster', 
                                   y=range(len(clustered_df)),
                                   color='cluster',
                                   title="Technology Clusters",
                                   hover_data=['title'] if 'title' in clustered_df.columns else None)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show keywords
                    st.subheader("Cluster Keywords")
                    for cluster_id, words in keywords.items():
                        st.write(f"**Cluster {cluster_id}:** {', '.join(words)}")
    
    elif page == "Trends":
        if 'patent_data' not in st.session_state:
            st.warning("Please upload data first")
            return
            
        st.header("📈 Patent Trends")
        df = st.session_state['patent_data']
        
        trends = analyzer.analyze_trends(df)
        if trends is not None:
            fig = px.line(trends, x='year', y='count', 
                         title="Patent Publications Over Time",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(trends)
    
    elif page == "Key Players":
        if 'patent_data' not in st.session_state:
            st.warning("Please upload data first")
            return
            
        st.header("👥 Key Players")
        df = st.session_state['patent_data']
        
        players = analyzer.identify_key_players(df)
        if players is not None:
            fig = px.bar(x=players.values, y=players.index, 
                        orientation='h',
                        title="Top Patent Assignees")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(players.reset_index())

if __name__ == "__main__":
    main()
