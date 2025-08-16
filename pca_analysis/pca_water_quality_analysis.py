#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) on Water Quality Data

This script performs PCA on water quality CSV files to analyze patterns
and trends in multiple water quality parameters across different zones.

Author: Generated for water quality analysis
Date: 2024
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Define input and output directories
INPUT_DATA_FOLDER = '../EPD_data/'
OUTPUT_FOLDER = './outputs/'
PCA_READY_DATA_FOLDER = './outputs/pca_ready_data/'
PLOTS_FOLDER = './outputs/plots/'

# Define the specific feature columns for PCA analysis
FEATURE_COLUMNS = [
    'Temperature (°C)',
    'Chlorophyll-a (μg/L)',
    'Dissolved Oxygen (%saturation)',
    'Salinity (psu)',
    'pH',
    '5-day Biochemical Oxygen Demand (mg/L)',
    'Suspended Solids (mg/L)',
    'Total Nitrogen (mg/L)',
    'Total Inorganic Nitrogen (mg/L)',
    'Ammonia Nitrogen (mg/L)',
    'Orthophosphate Phosphorus (mg/L)',
    'Total Phosphorus (mg/L)'
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_zone_name(filename):
    """
    Extract a clean zone name from the CSV filename.
    
    Args:
        filename (str): The CSV filename
        
    Returns:
        str: Clean zone name
    """
    # Remove path and extension, then clean up the name
    base_name = os.path.basename(filename)
    zone_name = os.path.splitext(base_name)[0]
    
    # Remove common prefixes/suffixes and clean up
    zone_name = zone_name.replace('_data', '').replace('water_quality_', '')
    zone_name = zone_name.replace('_', ' ').title()
    
    return zone_name

def load_and_preprocess_data(filepath):
    """
    Load CSV data and perform initial preprocessing.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with Date as index
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Parse the Date column and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def resample_to_monthly(df):
    """
    Resample data to monthly averages.
    
    Args:
        df (pd.DataFrame): Input DataFrame with Date index
        
    Returns:
        pd.DataFrame: Monthly averaged DataFrame
    """
    # Resample to monthly averages, only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    monthly_df = df[numeric_cols].resample('M').mean()
    
    return monthly_df

def prepare_pca_data(df, feature_columns):
    """
    Prepare data for PCA by imputing missing values and standardizing.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        feature_columns (list): List of feature column names
        
    Returns:
        tuple: (pca_ready_df, scaler, imputer)
    """
    # Filter to only include available feature columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    if len(available_features) == 0:
        print("Warning: No feature columns found in the data!")
        return None, None, None
    
    # Extract feature data
    feature_data = df[available_features].copy()
    
    # Step 1: Impute missing values using mean strategy
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(feature_data)
    
    # Step 2: Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(imputed_data)
    
    # Create DataFrame with standardized data
    pca_ready_df = pd.DataFrame(
        standardized_data,
        index=feature_data.index,
        columns=available_features
    )
    
    return pca_ready_df, scaler, imputer

def perform_pca_analysis(pca_ready_df):
    """
    Perform PCA analysis on the prepared data.
    
    Args:
        pca_ready_df (pd.DataFrame): Standardized data ready for PCA
        
    Returns:
        tuple: (pca_model, loadings_df, components_df)
    """
    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(pca_ready_df)
    
    # Create loadings DataFrame (feature contributions to each PC)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=pca_ready_df.columns,
        columns=['PC1', 'PC2']
    )
    
    # Create components DataFrame (time series of PC values)
    components_df = pd.DataFrame(
        pca_components,
        index=pca_ready_df.index,
        columns=['PC1', 'PC2']
    )
    
    return pca, loadings_df, components_df

def create_pc1_plot(components_df, zone_name, output_path):
    """
    Create a time series plot for PC1 with rolling mean.
    
    Args:
        components_df (pd.DataFrame): DataFrame with PC1 and PC2 values
        zone_name (str): Name of the zone for plot title
        output_path (str): Path to save the plot
    """
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Plot PC1 time series
    plt.plot(components_df.index, components_df['PC1'], 
             alpha=0.7, linewidth=1, label='PC1', color='steelblue')
    
    # Calculate and plot 12-month rolling mean
    rolling_mean = components_df['PC1'].rolling(window=12, center=True).mean()
    plt.plot(components_df.index, rolling_mean, 
             linewidth=2, label='12-month rolling mean', color='red')
    
    # Customize plot
    plt.title(f'Trend of PC1 for {zone_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('PC1 Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_loadings(loadings_df, zone_name):
    """
    Analyze PCA loadings to identify top positive and negative correlations.
    
    Args:
        loadings_df (pd.DataFrame): DataFrame with PCA loadings
        zone_name (str): Name of the zone
        
    Returns:
        dict: Dictionary with analysis results
    """
    # Sort loadings for PC1
    pc1_loadings = loadings_df['PC1'].sort_values(ascending=False)
    
    # Get top 3 positive and negative correlations
    top_positive = pc1_loadings.head(3).index.tolist()
    top_negative = pc1_loadings.tail(3).index.tolist()
    
    # Calculate explained variance ratio for PC1
    # Note: This will be calculated in the main loop where we have access to the PCA object
    
    return {
        'zone_name': zone_name,
        'top_positive_params': top_positive,
        'top_negative_params': top_negative,
        'pc1_loadings': pc1_loadings
    }

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """
    Main function to execute the complete PCA analysis workflow.
    """
    print("Starting PCA Analysis on Water Quality Data")
    print("=" * 50)
    
    # Initialize results collection
    all_results = []
    
    # Get all CSV files from input directory
    csv_pattern = os.path.join(INPUT_DATA_FOLDER, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {INPUT_DATA_FOLDER}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for i, filepath in enumerate(csv_files, 1):
        print(f"\nProcessing file {i}/{len(csv_files)}: {os.path.basename(filepath)}")
        
        # Extract zone name from filename
        zone_name = extract_zone_name(filepath)
        print(f"Zone: {zone_name}")
        
        # Load and preprocess data
        df = load_and_preprocess_data(filepath)
        if df is None:
            print(f"Skipping {filepath} due to loading error")
            continue
        
        # Resample to monthly averages
        monthly_df = resample_to_monthly(df)
        print(f"Data shape after monthly resampling: {monthly_df.shape}")
        
        # Prepare data for PCA
        pca_ready_df, scaler, imputer = prepare_pca_data(monthly_df, FEATURE_COLUMNS)
        if pca_ready_df is None:
            print(f"Skipping {zone_name} due to data preparation error")
            continue
        
        print(f"PCA-ready data shape: {pca_ready_df.shape}")
        
        # Save PCA-ready data
        pca_ready_filename = f"pca_ready_{zone_name.lower().replace(' ', '_')}.csv"
        pca_ready_path = os.path.join(PCA_READY_DATA_FOLDER, pca_ready_filename)
        pca_ready_df.to_csv(pca_ready_path)
        print(f"Saved PCA-ready data: {pca_ready_filename}")
        
        # Perform PCA analysis
        pca_model, loadings_df, components_df = perform_pca_analysis(pca_ready_df)
        print(f"PCA completed. Explained variance ratio: PC1={pca_model.explained_variance_ratio_[0]:.3f}, PC2={pca_model.explained_variance_ratio_[1]:.3f}")
        
        # Create PC1 visualization
        plot_filename = f"pc1_trend_{zone_name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
        create_pc1_plot(components_df, zone_name, plot_path)
        print(f"Saved plot: {plot_filename}")
        
        # Analyze loadings
        loading_analysis = analyze_loadings(loadings_df, zone_name)
        
        # Collect results for final summary
        zone_results = {
            'Zone Name': zone_name,
            'PC1 Explained Variance Ratio': pca_model.explained_variance_ratio_[0],
            'Top 3 Positive Correlations PC1': ', '.join(loading_analysis['top_positive_params']),
            'Top 3 Negative Correlations PC1': ', '.join(loading_analysis['top_negative_params']),
            'Data Points Used': len(pca_ready_df),
            'Features Available': len(pca_ready_df.columns)
        }
        
        all_results.append(zone_results)
        print(f"Analysis completed for {zone_name}")
    
    # Create final summary DataFrame
    print(f"\nCreating final summary report...")
    summary_df = pd.DataFrame(all_results)
    
    # Save summary to CSV
    summary_path = os.path.join(OUTPUT_FOLDER, 'pca_summary_all_zones.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Display summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(summary_df.to_string(index=False))
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nPCA analysis complete for all {len(csv_files)} zones.")
    
    return summary_df

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PCA_READY_DATA_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    
    # Run the main analysis
    try:
        results = main()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()