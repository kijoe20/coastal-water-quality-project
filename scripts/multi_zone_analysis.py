"""
Multi-Zone Water Quality Analysis Script

This script processes all 10 water quality CSV files and generates:
1. Time series plots for each parameter in each zone
2. Master table of Mann-Kendall test results for all zones
3. Master table of correlation results for all zones
4. Summary statistics and visualizations

Author: CIVL7009 Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
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

class MultiZoneWaterQualityAnalyzer:
    def __init__(self, data_dir="EPD_data", output_dir="multi_zone_results"):
        """
        Initialize the analyzer with data and output directories.
        
        Args:
            data_dir (str): Directory containing the CSV files
            output_dir (str): Directory to save results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different outputs
        (self.output_dir / "plots").mkdir(exist_ok=True)
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
        
        Args:
            csv_file (Path): Path to the CSV file
            
        Returns:
            tuple: (zone_name, processed_dataframe)
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
        
        Args:
            df (DataFrame): Dataframe with time series data
            parameter (str): Parameter name
            
        Returns:
            dict: Results dictionary
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
        
        Args:
            df (DataFrame): Dataframe with water quality data
            
        Returns:
            DataFrame: Correlation matrix
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
    
    def create_time_series_plots(self, zone_name, df):
        """
        Create time series plots for all parameters in a zone.
        
        Args:
            zone_name (str): Name of the zone
            df (DataFrame): Dataframe with time series data
        """
        # Get surface water data only for consistency
        surface_data = df[df['Depth'] == 'Surface Water'].copy()
        surface_data = surface_data.sort_values('Dates')
        
        # Create subplots for each parameter
        for param in self.parameters:
            if param not in surface_data.columns:
                continue
                
            # Clean data for this parameter
            param_data = surface_data[['Dates', param]].dropna()
            
            if len(param_data) < 5:  # Skip if too few data points
                continue
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(param_data['Dates'], param_data[param], 'o-', alpha=0.7, linewidth=1)
            plt.title(f'{param} - Zone {zone_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(param, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            safe_param_name = param.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
            plot_filename = f"zone_{zone_name}_{safe_param_name}_timeseries.png"
            plt.savefig(self.output_dir / "plots" / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_correlation_heatmap(self, zone_name, corr_matrix):
        """
        Create correlation heatmap for a zone.
        
        Args:
            zone_name (str): Name of the zone
            corr_matrix (DataFrame): Correlation matrix
        """
        if corr_matrix.empty:
            return
            
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title(f'Correlation Matrix - Zone {zone_name}', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"zone_{zone_name}_correlation_heatmap.png"
        plt.savefig(self.output_dir / "plots" / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """
        Run the complete analysis for all zones.
        """
        print("Starting multi-zone water quality analysis...")
        
        # Initialize results storage
        mk_results = []
        correlation_results = []
        
        # Process each CSV file
        for csv_file in self.csv_files:
            zone_name, df = self.load_and_preprocess_data(csv_file)
            
            # Create time series plots
            self.create_time_series_plots(zone_name, df)
            
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
                self.create_correlation_heatmap(zone_name, corr_matrix)
                
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
        
        Args:
            mk_results (list): List of MK test results for each zone
            correlation_results (list): List of correlation results
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
    
    def create_summary_statistics(self, mk_df, corr_df):
        """
        Create summary statistics and visualizations.
        
        Args:
            mk_df (DataFrame): MK test results dataframe
            corr_df (DataFrame): Correlation results dataframe
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
            
            # Create visualization of significant trends
            plt.figure(figsize=(15, 8))
            trend_counts = trends_df.groupby(['Zone', 'Trend']).size().unstack(fill_value=0)
            trend_counts.plot(kind='bar', stacked=True)
            plt.title('Significant Trends by Zone', fontsize=14, fontweight='bold')
            plt.xlabel('Zone')
            plt.ylabel('Number of Parameters')
            plt.xticks(rotation=45)
            plt.legend(title='Trend Direction')
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "significant_trends_by_zone.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Summary of strong correlations
        strong_correlations = corr_df[abs(corr_df['Correlation']) > 0.7].copy()
        if not strong_correlations.empty:
            strong_correlations.to_csv(self.output_dir / "tables" / "strong_correlations_summary.csv", index=False)
            
            # Create correlation strength visualization
            plt.figure(figsize=(12, 8))
            corr_by_zone = corr_df.groupby('Zone')['Correlation'].agg(['mean', 'std', 'count'])
            corr_by_zone['mean'].plot(kind='bar', yerr=corr_by_zone['std'])
            plt.title('Average Correlation Strength by Zone', fontsize=14, fontweight='bold')
            plt.xlabel('Zone')
            plt.ylabel('Average Correlation Coefficient')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "average_correlation_by_zone.png", dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function to run the analysis."""
    analyzer = MultiZoneWaterQualityAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 