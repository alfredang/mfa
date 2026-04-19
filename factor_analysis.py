"""
Multi-Factor Analysis Module
Using scikit-learn Factor Analysis
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import io


def load_sample_data(filepath="creditloandata.csv"):
    """Load the sample credit loan data"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise Exception(f"Error loading sample data: {e}")


def preprocess_data(df):
    """Preprocess data for factor analysis"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove customer_id as it's not a feature
    if 'customer_id' in numeric_cols:
        numeric_cols.remove('customer_id')
    
    # Create a copy with only numeric columns
    df_numeric = df[numeric_cols].copy()
    
    # Handle missing values
    df_numeric = df_numeric.fillna(df_numeric.mean())
    
    # Remove columns with zero variance
    variance = df_numeric.var()
    cols_to_keep = variance[variance > 0].index.tolist()
    df_numeric = df_numeric[cols_to_keep]
    
    return df_numeric, cols_to_keep


def perform_factor_analysis(X, n_components):
    """Perform Factor Analysis"""
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform Factor Analysis
    fa = FactorAnalysis(n_components=n_components, random_state=42)
    fa.fit(X_scaled)
    
    return fa, X_scaled


def get_correlation_matrix(df):
    """Get correlation matrix"""
    return df.corr()


def get_factor_loadings(fa, feature_names):
    """Get factor loadings from fitted model"""
    return fa.components_.T


def get_factor_scores(fa, X_scaled):
    """Get factor scores for samples"""
    return fa.transform(X_scaled)


def calculate_variance_explained(loadings):
    """Calculate variance explained by each factor"""
    eigenvalues = np.sum(loadings ** 2, axis=0)
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance * 100
    cumulative_variance = np.cumsum(explained_variance)
    
    return eigenvalues, explained_variance, cumulative_variance


def export_factor_loadings(loadings, feature_names, n_components):
    """Export factor loadings to CSV"""
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'Factor_{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    csv_buffer = io.StringIO()
    loadings_df.to_csv(csv_buffer)
    return csv_buffer.getvalue()


def export_factor_scores(scores, n_components):
    """Export factor scores to CSV"""
    scores_df = pd.DataFrame(
        scores,
        columns=[f'Factor_{i+1}' for i in range(n_components)]
    )
    
    scores_buffer = io.StringIO()
    scores_df.to_csv(scores_buffer, index_label='Sample')
    return scores_buffer.getvalue()


def interpret_factors(loadings, feature_names, n_components):
    """Interpret each factor based on top contributing variables"""
    interpretations = []
    
    for i in range(n_components):
        sorted_loadings = np.abs(loadings[:, i])
        top_indices = np.argsort(sorted_loadings)[-3:][::-1]
        top_features = [feature_names[j] for j in top_indices]
        interpretations.append({
            'factor': f'Factor {i+1}',
            'top_features': top_features
        })
    
    return interpretations