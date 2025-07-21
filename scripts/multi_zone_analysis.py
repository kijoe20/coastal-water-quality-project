import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import seaborn as sns
import os
import warnings
from pathlib import Path
import sys

# Add the scripts directory to the path to import analysis_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis_utils import mann_kendall_test, sen_slope

# Set up plotting style and suppress warnings
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Constants
DATA_DIR = '../EPD_data'
STATION_LOC_PATH = '../EPD_data/station_locations.csv'
OUTPUT_DIR = '../reports/figures'
TABLES_DIR = '../reports/tables'

KEY_PARAMETERS = [
    '5-day Biochemical Oxygen Demand (mg/L)', 'Suspended Solids (mg/L)',
    'Total Phosphorus (mg/L)', 'Total Nitrogen (mg/L)',
    'Ammonia Nitrogen (mg/L)', 'Orthophosphate Phosphorus (mg/L)',
    'Total Inorganic Nitrogen (mg/L)', 'Chlorophyll-a (μg/L)',
    'Dissolved Oxygen (%saturation)', 'Dissolved Oxygen (mg/L)',
    'E. coli (cfu/100mL)', 'Turbidity (NTU)', 'pH', 'Temperature (°C)',
    'Salinity (psu)'
]

class MultiZoneAnalyzer:
    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, tables_dir=TABLES_DIR):
        """Initialize the multi-zone analyzer."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.tables_dir = Path(tables_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all zone CSV files (1-10)
        self.zone_files = []
        for i in range(1, 11):
            zone_file = self.data_dir / f"{i}_marine_water_quality.csv"
            if zone_file.exists():
                self.zone_files.append(zone_file)
        
        print(f"Found {len(self.zone_files)} zone CSV files to process")
        
        # Load station locations
        self.station_locations = self.load_station_locations()

    def load_station_locations(self):
        """Load station location data."""
        try:
            return pd.read_csv(STATION_LOC_PATH)
        except FileNotFoundError:
            print("Warning: Station locations file not found.")
            return None

    def load_and_merge_data(self, zone_file):
        """Load water quality data for a zone and merge with station locations."""
        try:
            df_wq = pd.read_csv(zone_file)
            zone_num = zone_file.stem.split('_')[0]  # Extract zone number
            print(f"Successfully loaded Zone {zone_num} data: {len(df_wq)} records")
        except FileNotFoundError:
            print(f"Error: Zone file {zone_file} not found.")
            return None, None

        # Merge with station locations if available
        if self.station_locations is not None:
            merged_df = pd.merge(df_wq, self.station_locations, on='Station', how='left')
            
            # Report missing location data
            missing_location = merged_df[merged_df['Lat'].isnull()]['Station'].unique()
            if missing_location.size > 0:
                print(f"Zone {zone_num}: {len(missing_location)} stations missing location data")
        else:
            merged_df = df_wq
            zone_num = zone_file.stem.split('_')[0]

        return merged_df, zone_num

    def clean_data(self, df, zone_num):
        """Clean the dataframe for analysis."""
        # Convert 'Dates' column to datetime
        df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
        df.dropna(subset=['Dates'], inplace=True)
        df.set_index('Dates', inplace=True)

        # Convert key parameter columns to numeric, handling special values
        for col in KEY_PARAMETERS:
            if col in df.columns:
                # Replace common non-numeric values with NaN
                df[col] = df[col].replace(['<0.5', '>8.8', 'N/A', '<', '>', 'ND'], np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"Zone {zone_num}: Data cleaned. NaN counts:")
        nan_counts = df[KEY_PARAMETERS].isnull().sum()
        for param, count in nan_counts.items():
            if count > 0:
                print(f"  {param}: {count}")

        return df

    def calculate_correlation_matrix(self, df, zone_num):
        """Calculate correlation matrix for water quality parameters."""
        # Use surface water data only for consistency
        surface_data = df[df['Depth'] == 'Surface Water'] if 'Depth' in df.columns else df
        
        # Select parameters that exist in the data
        available_params = [param for param in KEY_PARAMETERS if param in surface_data.columns]
        
        if len(available_params) < 2:
            print(f"Zone {zone_num}: Insufficient parameters for correlation analysis")
            return None
        
        # Calculate correlation matrix
        corr_data = surface_data[available_params].dropna()
        
        if len(corr_data) < 10:
            print(f"Zone {zone_num}: Insufficient data points for correlation analysis")
            return None
        
        correlation_matrix = corr_data.corr()
        return correlation_matrix

    def plot_correlation_heatmap(self, correlation_matrix, zone_num):
        """Create and save correlation heatmap for a zone."""
        if correlation_matrix is None:
            return
        
        plt.figure(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
        heatmap = sns.heatmap(correlation_matrix, 
                            mask=mask,
                            annot=True, 
                            cmap='RdBu_r', 
                            center=0,
                            square=True,
                            fmt='.2f',
                            cbar_kws={"shrink": .8},
                            annot_kws={'size': 8})
        
        plt.title(f'Water Quality Parameters Correlation Matrix - Zone {zone_num}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        filename = f'correlation_heatmap_zone_{zone_num}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Zone {zone_num}: Correlation heatmap saved to {filepath}")
        plt.close()

    def calculate_mann_kendall_trends(self, df, zone_num):
        """Calculate Mann-Kendall trend tests for all parameters in a zone."""
        # Use surface water data only
        surface_data = df[df['Depth'] == 'Surface Water'] if 'Depth' in df.columns else df
        
        mk_results = {'Zone': zone_num}
        
        for param in KEY_PARAMETERS:
            if param not in surface_data.columns:
                mk_results[f'{param}_mk_pvalue'] = np.nan
                mk_results[f'{param}_sen_slope'] = np.nan
                mk_results[f'{param}_trend'] = 'No Data'
                mk_results[f'{param}_n_samples'] = 0
                continue
            
            # Get time series data
            param_data = surface_data[param].dropna()
            
            if len(param_data) < 10:  # Need minimum samples
                mk_results[f'{param}_mk_pvalue'] = np.nan
                mk_results[f'{param}_sen_slope'] = np.nan
                mk_results[f'{param}_trend'] = 'Insufficient Data'
                mk_results[f'{param}_n_samples'] = len(param_data)
                continue
            
            try:
                # Calculate Mann-Kendall test
                mk_pvalue = mann_kendall_test(param_data.values)
                sen_slope_val = sen_slope(param_data.values)
                
                # Determine trend significance and direction
                if mk_pvalue < 0.05:
                    trend = 'Increasing' if sen_slope_val > 0 else 'Decreasing'
                else:
                    trend = 'No Trend'
                
                mk_results[f'{param}_mk_pvalue'] = mk_pvalue
                mk_results[f'{param}_sen_slope'] = sen_slope_val
                mk_results[f'{param}_trend'] = trend
                mk_results[f'{param}_n_samples'] = len(param_data)
                
            except Exception as e:
                print(f"Zone {zone_num}, {param}: Error in MK test - {e}")
                mk_results[f'{param}_mk_pvalue'] = np.nan
                mk_results[f'{param}_sen_slope'] = np.nan
                mk_results[f'{param}_trend'] = 'Error'
                mk_results[f'{param}_n_samples'] = len(param_data)
        
        return mk_results

    def extract_correlation_pairs(self, correlation_matrix, zone_num):
        """Extract correlation pairs from correlation matrix."""
        if correlation_matrix is None:
            return []
        
        correlation_pairs = []
        
        # Extract upper triangle (avoid duplicates)
        for i in range(len(correlation_matrix.index)):
            for j in range(i+1, len(correlation_matrix.columns)):
                param1 = correlation_matrix.index[i]
                param2 = correlation_matrix.columns[j]
                correlation_value = correlation_matrix.iloc[i, j]
                
                if not pd.isna(correlation_value):
                    correlation_pairs.append({
                        'Zone': zone_num,
                        'Parameter1': param1,
                        'Parameter2': param2,
                        'Correlation': correlation_value,
                        'Abs_Correlation': abs(correlation_value),
                        'Strength': self.classify_correlation_strength(correlation_value)
                    })
        
        return correlation_pairs

    def classify_correlation_strength(self, corr_value):
        """Classify correlation strength."""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.8:
            return 'Very Strong'
        elif abs_corr >= 0.6:
            return 'Strong'
        elif abs_corr >= 0.4:
            return 'Moderate'
        elif abs_corr >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'

    def plot_station_map(self, df, zone_num):
        """Create and save a map of station locations for a zone."""
        if 'Lon' not in df.columns or 'Lat' not in df.columns:
            print(f"Zone {zone_num}: Longitude/Latitude columns not found. Skipping map plot.")
            return

        unique_stations = df[['Lon', 'Lat', 'Station']].drop_duplicates()
        if len(unique_stations) == 0:
            print(f"Zone {zone_num}: No station location data available.")
            return

        gdf = gpd.GeoDataFrame(
            unique_stations,
            geometry=gpd.points_from_xy(unique_stations.Lon, unique_stations.Lat),
            crs="EPSG:4326"
        )

        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Try to use cartopy if available, otherwise use simple plot
        try:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([113.8, 114.5, 22.1, 22.6], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, edgecolor='black')
            ax.add_feature(cfeature.OCEAN)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            ax.scatter(gdf.Lon, gdf.Lat, transform=ccrs.PlateCarree(),
                      c='red', s=50, edgecolor='black', zorder=5)
        except:
            # Fallback to simple scatter plot
            ax.scatter(gdf.Lon, gdf.Lat, c='red', s=50, edgecolor='black')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        ax.set_title(f'Station Locations - Zone {zone_num}')
        
        filename = f'station_map_zone_{zone_num}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Zone {zone_num}: Station map saved to {filepath}")
        plt.close()

    def plot_monthly_time_series(self, df, zone_num):
        """Calculate monthly averages and plot time series for each parameter with rolling mean."""
        available_params = [param for param in KEY_PARAMETERS if param in df.columns]
        
        if not available_params:
            print(f"Zone {zone_num}: No parameters available for time series plot.")
            return

        # Filter for surface water data only for consistency with other analyses
        surface_data = df[df['Depth'] == 'Surface Water'] if 'Depth' in df.columns else df
        
        if len(surface_data) == 0:
            print(f"Zone {zone_num}: No surface water data available for time series plot.")
            return

        # Calculate monthly averages
        monthly_averages = surface_data[available_params].resample('ME').mean()
        
        # Check if we have sufficient data for meaningful analysis
        if len(monthly_averages) < 3:
            print(f"Zone {zone_num}: Insufficient monthly data points ({len(monthly_averages)}) for time series plot.")
            return
        
        # Rolling window parameter for trend visualization
        ROLLING_WINDOW = 6  # 6-month rolling mean for better trend visualization

        # Create multi-panel plot
        num_params = len(available_params)
        num_cols = 3
        num_rows = int(np.ceil(num_params / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, param in enumerate(available_params):
            # Skip parameters with insufficient data
            param_data = monthly_averages[param].dropna()
            if len(param_data) < 3:
                axes[i].text(0.5, 0.5, f'Insufficient data\nfor {param}', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[i].set_title(param, fontsize=10)
                continue
            
            # Plot original monthly averages
            monthly_averages[param].plot(ax=axes[i], label='Monthly Average', 
                                       alpha=0.7, linewidth=1)
            
            # Calculate rolling mean with proper error handling
            try:
                rolling_mean = monthly_averages[param].rolling(window=ROLLING_WINDOW, 
                                                             min_periods=1, center=True).mean()
                
                # Only plot rolling mean if we have valid data
                valid_rolling = rolling_mean.dropna()
                if len(valid_rolling) > 0:
                    rolling_mean.plot(ax=axes[i], 
                                    label=f'{ROLLING_WINDOW}-Month Rolling Mean', 
                                    color='red', linewidth=2)
                else:
                    print(f"Zone {zone_num}, {param}: Rolling mean calculation resulted in no valid data.")
                    
            except Exception as e:
                print(f"Zone {zone_num}, {param}: Error calculating rolling mean - {e}")
                # Continue without rolling mean for this parameter
            
            axes[i].set_title(param, fontsize=10)
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Monthly Average')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'Monthly Averages of Water Quality Parameters with Rolling Mean - Zone {zone_num}', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        filename = f'monthly_timeseries_zone_{zone_num}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Zone {zone_num}: Time series plot saved to {filepath}")
        plt.close()

    def create_master_tables(self, all_mk_results, all_correlation_pairs):
        """Create and save master tables for MK test and correlation results."""
        # Create Mann-Kendall master table
        mk_df = pd.DataFrame(all_mk_results)
        mk_filepath = self.tables_dir / 'master_mk_test_results.csv'
        mk_df.to_csv(mk_filepath, index=False)
        print(f"Master MK test results saved to {mk_filepath}")

        # Create correlation master table
        if all_correlation_pairs:
            corr_df = pd.DataFrame(all_correlation_pairs)
            corr_filepath = self.tables_dir / 'master_correlation_results.csv'
            corr_df.to_csv(corr_filepath, index=False)
            print(f"Master correlation results saved to {corr_filepath}")
            
            # Create summary tables
            self.create_summary_tables(mk_df, corr_df)
        else:
            print("No correlation data available for master table.")

    def create_summary_tables(self, mk_df, corr_df):
        """Create summary tables for significant trends and strong correlations."""
        # Summary of significant trends (p < 0.05)
        significant_trends = []
        
        for _, row in mk_df.iterrows():
            zone = row['Zone']
            for param in KEY_PARAMETERS:
                pvalue_col = f'{param}_mk_pvalue'
                slope_col = f'{param}_sen_slope'
                trend_col = f'{param}_trend'
                
                if pvalue_col in row and not pd.isna(row[pvalue_col]):
                    if row[pvalue_col] < 0.05:
                        significant_trends.append({
                            'Zone': zone,
                            'Parameter': param,
                            'P_value': row[pvalue_col],
                            'Sen_Slope': row[slope_col] if slope_col in row else np.nan,
                            'Trend': row[trend_col] if trend_col in row else 'Unknown',
                            'Significance': 'Highly Significant' if row[pvalue_col] < 0.01 else 'Significant'
                        })
        
        if significant_trends:
            trends_df = pd.DataFrame(significant_trends)
            trends_filepath = self.tables_dir / 'significant_trends_summary.csv'
            trends_df.to_csv(trends_filepath, index=False)
            print(f"Significant trends summary saved to {trends_filepath}")
            print(f"Found {len(significant_trends)} significant trends across all zones")

        # Summary of strong correlations (|r| > 0.7)
        if not corr_df.empty:
            strong_correlations = corr_df[corr_df['Abs_Correlation'] > 0.7].copy()
            if not strong_correlations.empty:
                strong_corr_filepath = self.tables_dir / 'strong_correlations_summary.csv'
                strong_correlations.to_csv(strong_corr_filepath, index=False)
                print(f"Strong correlations summary saved to {strong_corr_filepath}")
                print(f"Found {len(strong_correlations)} strong correlations across all zones")

    def run_complete_analysis(self):
        """Run the complete multi-zone analysis."""
        print("Starting comprehensive multi-zone water quality analysis...")
        print("=" * 60)
        
        all_mk_results = []
        all_correlation_pairs = []
        
        # Process each zone
        for zone_file in self.zone_files:
            print(f"\nProcessing {zone_file.name}...")
            print("-" * 40)
            
            # Load and merge data
            merged_df, zone_num = self.load_and_merge_data(zone_file)
            if merged_df is None:
                continue
            
            # Clean data
            cleaned_df = self.clean_data(merged_df.copy(), zone_num)
            
            # Display basic statistics
            print(f"Zone {zone_num}: {len(cleaned_df)} records after cleaning")
            
            # Calculate Mann-Kendall trends
            mk_results = self.calculate_mann_kendall_trends(cleaned_df, zone_num)
            all_mk_results.append(mk_results)
            
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(cleaned_df, zone_num)
            
            # Extract correlation pairs
            correlation_pairs = self.extract_correlation_pairs(correlation_matrix, zone_num)
            all_correlation_pairs.extend(correlation_pairs)
            
            # Generate plots
            self.plot_correlation_heatmap(correlation_matrix, zone_num)
            self.plot_station_map(cleaned_df, zone_num)
            self.plot_monthly_time_series(cleaned_df, zone_num)
            
            print(f"Zone {zone_num}: Analysis completed")
        
        # Create master tables
        print("\n" + "=" * 60)
        print("Creating master tables...")
        self.create_master_tables(all_mk_results, all_correlation_pairs)
        
        print("\n" + "=" * 60)
        print("Multi-zone analysis completed successfully!")
        print(f"Results saved to:")
        print(f"  - Figures: {self.output_dir}")
        print(f"  - Tables: {self.tables_dir}")

def main():
    """Main function to run the analysis."""
    analyzer = MultiZoneAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == '__main__':
    main()

