"""
Simplified Multi-Zone Water Quality Analysis Script

This script processes all 10 water quality CSV files and generates:
1. Master table of Mann-Kendall test results for all zones
2. Master table of correlation results for all zones
3. Summary statistics

Focuses on generating the master tables efficiently.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pathlib import Path
import sys

# Add the scripts directory to the path to import analysis_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis_utils import mann_kendall_test, sen_slope

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class SimpleMultiZoneAnalyzer:
    def __init__(self, data_dir="EPD_data", output_dir="multi_zone_results"):
        """
        Initialize the analyzer with data and output directories.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different outputs
        (self.output_dir / "tables").mkdir(exist_ok=True)
        
        # Define water quality parameters
        self.parameters = [
            "5-day Biochemical Oxygen Demand (mg/L)",
            "Suspended Solids (mg/L)", 
            "Total Phosphorus (mg/L)",
            "Total Nitrogen (mg/L)",
            "Ammonia Nitrogen (mg/L)",
            "Orthophosphate Phosphorus (mg/L)",
            "Total Inorganic Nitrogen (mg/L)",
            "Chlorophyll-a (μg/L)",
            "Dissolved Oxygen (%saturation)",
            "Dissolved Oxygen (mg/L)",
            "E. coli (cfu/100mL)",
            "Turbidity (NTU)",
            "pH",
            "Temperature (°C)",
            "Salinity (psu)"
        ]
        
        # Get list of CSV files
        self.csv_files = sorted([f for f in self.data_dir.glob("*.csv") 
                               if f.name.startswith(("1_", "2_", "3_", "4_", "5_", 
                                                    "6_", "7_", "8_", "9_", "10_"))])
        
        print(f"Found {len(self.csv_files)} CSV files to process")
        
    def load_and_preprocess_data(self, csv_file):
        """
        Load and preprocess a single CSV file.
        """
        print(f"Processing {csv_file.name}...")
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Extract zone name from filename
        zone_name = csv_file.stem.replace("_marine_water_quality", "")
        
        # Convert date column
        df['Dates'] = pd.to_datetime(df['Dates'])
        
        # Handle missing values and non-numeric data
        for param in self.parameters:
            if param in df.columns:
                # Replace '<0.5', '>8.8', 'N/A' etc. with NaN
                df[param] = pd.to_numeric(df[param].replace(['<0.5', '>8.8', 'N/A'], np.nan), errors='coerce')
        
        return zone_name, df
    
    def calculate_trend_analysis(self, df, parameter):
        """
        Calculate Mann-Kendall trend test and Sen's slope for a parameter.
        """
        if parameter not in df.columns:
            return {'mk_pvalue': np.nan, 'sen_slope': np.nan, 'n_samples': 0}
        
        # Get time series data (surface water only for consistency)
        surface_data = df[df['Depth'] == 'Surface Water'][parameter].dropna()
        
        if len(surface_data) < 10:  # Need minimum samples for meaningful analysis
            return {'mk_pvalue': np.nan, 'sen_slope': np.nan, 'n_samples': len(surface_data)}
        
        try:
            mk_pvalue = mann_kendall_test(surface_data.values)
            sen_slope_val = sen_slope(surface_data.values)
            
            return {
                'mk_pvalue': mk_pvalue,
                'sen_slope': sen_slope_val,
                'n_samples': len(surface_data)
            }
        except:
            return {'mk_pvalue': np.nan, 'sen_slope': np.nan, 'n_samples': len(surface_data)}
    
    def calculate_correlation_matrix(self, df):
        """
        Calculate correlation matrix for all parameters.
        """
        # Get surface water data only
        surface_data = df[df['Depth'] == 'Surface Water']
        
        # Select only numeric parameters
        numeric_params = [param for param in self.parameters if param in surface_data.columns]
        numeric_data = surface_data[numeric_params].dropna()
        
        if len(numeric_data) < 10:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        return corr_matrix
    
    def run_analysis(self):
        """
        Run the complete analysis for all zones.
        """
        print("Starting simplified multi-zone water quality analysis...")
        
        # Initialize results storage
        mk_results = []
        correlation_results = []
        
        # Process each CSV file
        for csv_file in self.csv_files:
            zone_name, df = self.load_and_preprocess_data(csv_file)
            
            # Calculate Mann-Kendall results for each parameter
            zone_mk_results = {'Zone': zone_name}
            for param in self.parameters:
                trend_result = self.calculate_trend_analysis(df, param)
                zone_mk_results[f'{param}_mk_pvalue'] = trend_result['mk_pvalue']
                zone_mk_results[f'{param}_sen_slope'] = trend_result['sen_slope']
                zone_mk_results[f'{param}_n_samples'] = trend_result['n_samples']
            
            mk_results.append(zone_mk_results)
            
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(df)
            if not corr_matrix.empty:
                # Store correlation results
                for i, param1 in enumerate(corr_matrix.index):
                    for j, param2 in enumerate(corr_matrix.columns):
                        if i < j:  # Only store upper triangle to avoid duplicates
                            correlation_results.append({
                                'Zone': zone_name,
                                'Parameter1': param1,
                                'Parameter2': param2,
                                'Correlation': corr_matrix.iloc[i, j]
                            })
        
        # Create master tables
        self.create_master_tables(mk_results, correlation_results)
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
    
    def create_master_tables(self, mk_results, correlation_results):
        """
        Create master tables for MK test and correlation results.
        """
        # Create MK results master table
        mk_df = pd.DataFrame(mk_results)
        mk_df.to_csv(self.output_dir / "tables" / "master_mk_test_results.csv", index=False)
        
        # Create correlation results master table
        corr_df = pd.DataFrame(correlation_results)
        corr_df.to_csv(self.output_dir / "tables" / "master_correlation_results.csv", index=False)
        
        # Create summary statistics
        self.create_summary_statistics(mk_df, corr_df)
        
        print("Master tables created successfully!")
        print(f"MK test results: {len(mk_df)} zones")
        print(f"Correlation results: {len(corr_df)} parameter pairs")
    
    def create_summary_statistics(self, mk_df, corr_df):
        """
        Create summary statistics and visualizations.
        """
        # Summary of significant trends
        significant_trends = []
        for zone in mk_df['Zone']:
            zone_data = mk_df[mk_df['Zone'] == zone]
            for param in self.parameters:
                pvalue_col = f'{param}_mk_pvalue'
                slope_col = f'{param}_sen_slope'
                
                if pvalue_col in zone_data.columns:
                    pvalue = zone_data[pvalue_col].iloc[0]
                    slope = zone_data[slope_col].iloc[0]
                    
                    if not pd.isna(pvalue) and pvalue < 0.05:
                        trend = "Increasing" if slope > 0 else "Decreasing"
                        significant_trends.append({
                            'Zone': zone,
                            'Parameter': param,
                            'P-value': pvalue,
                            'Sen Slope': slope,
                            'Trend': trend
                        })
        
        if significant_trends:
            trends_df = pd.DataFrame(significant_trends)
            trends_df.to_csv(self.output_dir / "tables" / "significant_trends_summary.csv", index=False)
            print(f"Found {len(significant_trends)} significant trends")
        
        # Summary of strong correlations
        strong_correlations = corr_df[abs(corr_df['Correlation']) > 0.7].copy()
        if not strong_correlations.empty:
            strong_correlations.to_csv(self.output_dir / "tables" / "strong_correlations_summary.csv", index=False)
            print(f"Found {len(strong_correlations)} strong correlations")

def main():
    """Main function to run the analysis."""
    analyzer = SimpleMultiZoneAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 