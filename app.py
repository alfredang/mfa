"""
Multi-Factor Analysis Web Tool
Using Streamlit and scikit-learn Factor Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the factor analysis module
from factor_analysis import (
    load_sample_data,
    preprocess_data,
    perform_factor_analysis,
    get_correlation_matrix,
    get_factor_loadings,
    get_factor_scores,
    calculate_variance_explained,
    export_factor_loadings,
    export_factor_scores,
    interpret_factors
)

# Page configuration
st.set_page_config(
    page_title="Multi-Factor Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .header {
        background-color: #2c3e50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def plot_correlation_matrix(corr):
    """Plot correlation matrix heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, ax=ax)
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_factor_loadings(loadings, feature_names, n_components):
    """Plot factor loadings as a heatmap"""
    loadings_df = pd.DataFrame(
        loadings, 
        columns=[f'Factor {i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, ax=ax)
    plt.title('Factor Loadings Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Factors', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    plt.tight_layout()
    return fig


def plot_scree_plot(explained_variance):
    """Plot scree plot for factor selection"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, len(explained_variance) + 1)
    ax.plot(x, explained_variance, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(x, explained_variance, alpha=0.3)
    ax.set_xlabel('Factor Number', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('Scree Plot - Explained Variance by Factor', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    return fig


def plot_factor_scores(scores, n_components):
    """Plot factor scores scatter plot"""
    if n_components >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(scores[:, 0], scores[:, 1], 
                            c=range(len(scores)), cmap='viridis', 
                            alpha=0.6, s=50)
        ax.set_xlabel('Factor 1', fontsize=12)
        ax.set_ylabel('Factor 2', fontsize=12)
        ax.set_title('Factor Scores (Factor 1 vs Factor 2)', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        plt.tight_layout()
        return fig
    return None


def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>📊 Multi-Factor Analysis Tool</h1>
        <p>Perform Factor Analysis on your data using scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for data input
    st.sidebar.header("📁 Data Input")
    
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Load Sample Data (creditloandata.csv)", "Upload CSV File"]
    )
    
    df = None
    
    if data_source == "Load Sample Data (creditloandata.csv)":
        try:
            df = load_sample_data()
            st.sidebar.success("✅ Sample data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    
    if df is not None:
        # Main content
        st.header("📋 Data Preview")
        
        # Show data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
        
        # Show first few rows
        with st.expander("View Raw Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Preprocess data
        st.header("⚙️ Data Preprocessing")
        
        df_numeric, feature_names = preprocess_data(df)
        
        st.write(f"**Selected {len(feature_names)} numeric features for analysis:**")
        st.write(", ".join(feature_names))
        
        with st.expander("View Preprocessed Data"):
            st.dataframe(df_numeric.head(10), use_container_width=True)
        
        # Factor Analysis Parameters
        st.header("🔧 Factor Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_components = st.slider(
                "Number of Factors",
                min_value=2,
                max_value=min(10, len(feature_names)),
                value=min(3, len(feature_names)),
                help="Number of latent factors to extract"
            )
        
        with col2:
            st.info("Rotation options coming soon!")
        
        # Run Analysis
        if st.button("🚀 Run Factor Analysis", type="primary"):
            with st.spinner("Running Factor Analysis..."):
                try:
                    # Perform Factor Analysis
                    fa, X_scaled = perform_factor_analysis(df_numeric, n_components)
                    
                    # Get results
                    loadings = get_factor_loadings(fa, feature_names)
                    scores = get_factor_scores(fa, X_scaled)
                    
                    # Calculate explained variance
                    eigenvalues, explained_variance, cumulative_variance = calculate_variance_explained(loadings)
                    
                    st.success("✅ Factor Analysis completed!")
                    
                    # Results
                    st.header("📊 Analysis Results")
                    
                    # Explained Variance
                    st.subheader("Variance Explained")
                    
                    variance_df = pd.DataFrame({
                        'Factor': [f'Factor {i+1}' for i in range(n_components)],
                        'Eigenvalue': eigenvalues,
                        'Variance (%)': np.round(explained_variance, 2),
                        'Cumulative Variance (%)': np.round(cumulative_variance, 2)
                    })
                    st.dataframe(variance_df, use_container_width=True)
                    
                    # Visualizations
                    st.header("📈 Visualizations")
                    
                    # Tab layout for visualizations
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Correlation Matrix", 
                        "🔲 Factor Loadings", 
                        "📉 Scree Plot",
                        "🎯 Factor Scores"
                    ])
                    
                    with tab1:
                        st.subheader("Correlation Matrix")
                        corr = get_correlation_matrix(df_numeric)
                        fig1 = plot_correlation_matrix(corr)
                        st.pyplot(fig1)
                        
                    with tab2:
                        st.subheader("Factor Loadings")
                        st.write("Factor loadings show how each variable relates to each factor.")
                        fig2 = plot_factor_loadings(loadings, feature_names, n_components)
                        st.pyplot(fig2)
                        
                        # Interpretation
                        st.subheader("Factor Interpretation")
                        interpretations = interpret_factors(loadings, feature_names, n_components)
                        for interp in interpretations:
                            factor_num = int(interp['factor'].split()[-1]) - 1
                            st.write(f"**{interp['factor']}** (explains {explained_variance[factor_num]:.1f}% variance):")
                            st.write(f"   Top contributing variables: {', '.join(interp['top_features'])}")
                    
                    with tab3:
                        st.subheader("Scree Plot")
                        fig3 = plot_scree_plot(explained_variance)
                        st.pyplot(fig3)
                        st.info("💡 Look for the 'elbow' in the plot to determine the optimal number of factors.")
                    
                    with tab4:
                        st.subheader("Factor Scores")
                        if n_components >= 2:
                            fig4 = plot_factor_scores(scores, n_components)
                            if fig4:
                                st.pyplot(fig4)
                            st.info("💡 Each point represents a sample. Samples close together are similar in factor space.")
                        else:
                            st.warning("Need at least 2 factors to plot factor scores.")
                    
                    # Download results
                    st.header("💾 Download Results")
                    
                    # Download factor loadings
                    loadings_csv = export_factor_loadings(loadings, feature_names, n_components)
                    st.download_button(
                        label="📥 Download Factor Loadings",
                        data=loadings_csv,
                        file_name="factor_loadings.csv",
                        mime="text/csv"
                    )
                    
                    # Download factor scores
                    scores_csv = export_factor_scores(scores, n_components)
                    st.download_button(
                        label="📥 Download Factor Scores",
                        data=scores_csv,
                        file_name="factor_scores.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.info("Please check your data and try again.")
                
    else:
        st.info("👈 Please select a data source from the sidebar to begin.")
        
        # Show sample data structure
        st.header("📖 How to Use")
        st.markdown("""
        1. **Select Data Source**: Choose to load sample data or upload your own CSV
        2. **Preview Data**: View your data before analysis
        3. **Configure Settings**: Set the number of factors
        4. **Run Analysis**: Execute factor analysis
        5. **Interpret Results**: Review loadings, variance, and visualizations
        6. **Download**: Export results for further use
        """)


if __name__ == "__main__":
    main()
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; padding: 10px; color: #666;">'
        'Powered by <a href="https://www.tertiarycourses.com.sg" target="_blank">'
        'Tertiary Infotech Academy Pte Ltd</a>'
        '</div>',
        unsafe_allow_html=True
    )