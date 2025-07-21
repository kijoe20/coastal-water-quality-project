import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data():
    """Load and clean the water quality data"""
    # Load the water quality data
    df_wq = pd.read_csv('../EPD_data/1_marine_water_quality.csv')
    
    # Convert Dates column to datetime
    df_wq['Dates'] = pd.to_datetime(df_wq['Dates'], errors='coerce')
    
    # Drop rows where date conversion failed
    df_wq = df_wq.dropna(subset=['Dates'])
    
    # Convert the specified columns to numeric, handling 'N/A' and '<5' values
    columns_to_convert = ['E. coli (cfu/100mL)', '5-day Biochemical Oxygen Demand (mg/L)', 
                         'Ammonia Nitrogen (mg/L)', 'Total Inorganic Nitrogen (mg/L)', 
                         'Total Phosphorus (mg/L)', 'Suspended Solids (mg/L)']
    
    for col in columns_to_convert:
        if col in df_wq.columns:
            # Replace 'N/A' with NaN and handle '<5' values
            df_wq[col] = df_wq[col].replace('N/A', np.nan)
            df_wq[col] = df_wq[col].replace('<5', '2.5')  # Use midpoint for '<5'
            df_wq[col] = pd.to_numeric(df_wq[col], errors='coerce')
    
    return df_wq

def create_period_comparison(df):
    """Create pre-1995 and post-1995 subsets and calculate means"""
    
    # Create subsets for pre-1995 and post-1995
    pre_1995 = df[df['Dates'] < '1995-01-01']
    post_1995 = df[df['Dates'] >= '1995-01-01']
    
    # Define the columns to analyze (mapping to actual column names)
    columns_mapping = {
        'E. coli': 'E. coli (cfu/100mL)',
        'BOD5': '5-day Biochemical Oxygen Demand (mg/L)',
        'NH3-N': 'Ammonia Nitrogen (mg/L)',
        'TIN': 'Total Inorganic Nitrogen (mg/L)',
        'TP': 'Total Phosphorus (mg/L)',
        'SS': 'Suspended Solids (mg/L)'
    }
    
    # Calculate means for both periods
    results = {}
    
    for short_name, full_name in columns_mapping.items():
        if full_name in df.columns:
            pre_mean = pre_1995[full_name].mean()
            post_mean = post_1995[full_name].mean()
            difference = post_mean - pre_mean
            percent_change = ((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else np.nan
            
            results[short_name] = {
                'Pre-1995 Mean': pre_mean,
                'Post-1995 Mean': post_mean,
                'Difference': difference,
                'Percent Change (%)': percent_change
            }
    
    return results, pre_1995, post_1995

def display_comparison_table(results):
    """Display the comparison results in a formatted table"""
    
    # Create DataFrame for display
    comparison_df = pd.DataFrame(results).T
    
    # Round numeric values for better display
    numeric_columns = ['Pre-1995 Mean', 'Post-1995 Mean', 'Difference', 'Percent Change (%)']
    for col in numeric_columns:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(3)
    
    print("=" * 80)
    print("WATER QUALITY COMPARISON: PRE-1995 vs POST-1995")
    print("=" * 80)
    print(comparison_df.to_string())
    print("=" * 80)
    
    return comparison_df

def print_summary_stats(pre_1995, post_1995):
    """Print summary statistics for both periods"""
    
    print(f"\nDATA SUMMARY:")
    print(f"Pre-1995 period: {len(pre_1995)} records")
    print(f"Post-1995 period: {len(post_1995)} records")
    print(f"Total records: {len(pre_1995) + len(post_1995)}")
    
    if len(pre_1995) > 0:
        print(f"Pre-1995 date range: {pre_1995['Dates'].min()} to {pre_1995['Dates'].max()}")
    if len(post_1995) > 0:
        print(f"Post-1995 date range: {post_1995['Dates'].min()} to {post_1995['Dates'].max()}")

def main():
    """Main function to run the analysis"""
    
    print("Loading and cleaning water quality data...")
    df = load_and_clean_data()
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['Dates'].min()} to {df['Dates'].max()}")
    
    print("\nCreating pre-1995 and post-1995 subsets...")
    results, pre_1995, post_1995 = create_period_comparison(df)
    
    print_summary_stats(pre_1995, post_1995)
    
    print("\nCalculating comparison table...")
    comparison_df = display_comparison_table(results)
    
    # Save results to CSV
    comparison_df.to_csv('water_quality_comparison_results.csv')
    print(f"\nResults saved to 'water_quality_comparison_results.csv'")
    
    return comparison_df, pre_1995, post_1995

if __name__ == "__main__":
    main() 