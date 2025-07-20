import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the water quality data"""
    # Load the water quality data
    df_wq = pd.read_csv('EPD_data/2_marine_water_quality.csv')
    
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

def perform_statistical_tests(pre_1995, post_1995, columns_mapping):
    """Perform statistical tests to compare the two periods"""
    
    results = {}
    
    for short_name, full_name in columns_mapping.items():
        if full_name in pre_1995.columns:
            pre_data = pre_1995[full_name].dropna()
            post_data = post_1995[full_name].dropna()
            
            if len(pre_data) > 0 and len(post_data) > 0:
                # Perform Mann-Whitney U test (non-parametric)
                try:
                    statistic, p_value = stats.mannwhitneyu(pre_data, post_data, alternative='two-sided')
                    results[short_name] = {
                        'Mann-Whitney U': statistic,
                        'p-value': p_value,
                        'Significant': p_value < 0.05,
                        'Pre-1995 n': len(pre_data),
                        'Post-1995 n': len(post_data)
                    }
                except:
                    results[short_name] = {
                        'Mann-Whitney U': np.nan,
                        'p-value': np.nan,
                        'Significant': False,
                        'Pre-1995 n': len(pre_data),
                        'Post-1995 n': len(post_data)
                    }
    
    return results

def create_visualizations(pre_1995, post_1995, columns_mapping):
    """Create visualizations for the comparison"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Water Quality Parameters: Pre-1995 vs Post-1995 Comparison', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for i, (short_name, full_name) in enumerate(columns_mapping.items()):
        if i < 6 and full_name in pre_1995.columns:
            ax = axes[i]
            
            # Prepare data
            pre_data = pre_1995[full_name].dropna()
            post_data = post_1995[full_name].dropna()
            
            # Create box plots
            data_to_plot = [pre_data, post_data]
            labels = ['Pre-1995', 'Post-1995']
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{short_name}', fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Add sample size annotations
            ax.text(0.02, 0.98, f'n={len(pre_data)}', transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10)
            ax.text(0.98, 0.98, f'n={len(post_data)}', transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right', fontsize=10)
    
    # Remove empty subplots
    for i in range(len(columns_mapping), 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('water_quality_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(comparison_df, stats_results):
    """Print detailed analysis results"""
    
    print("=" * 100)
    print("DETAILED WATER QUALITY ANALYSIS: PRE-1995 vs POST-1995")
    print("=" * 100)
    
    print("\nCOMPARISON TABLE:")
    print(comparison_df.to_string())
    
    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE TESTS (Mann-Whitney U Test)")
    print("=" * 100)
    
    stats_df = pd.DataFrame(stats_results).T
    print(stats_df.to_string())
    
    print("\n" + "=" * 100)
    print("KEY FINDINGS:")
    print("=" * 100)
    
    for param in comparison_df.index:
        pre_mean = comparison_df.loc[param, 'Pre-1995 Mean']
        post_mean = comparison_df.loc[param, 'Post-1995 Mean']
        diff = comparison_df.loc[param, 'Difference']
        pct_change = comparison_df.loc[param, 'Percent Change (%)']
        
        if param in stats_results:
            significant = stats_results[param]['Significant']
            p_value = stats_results[param]['p-value']
            
            print(f"\n{param}:")
            print(f"  Pre-1995 Mean: {pre_mean:.3f}")
            print(f"  Post-1995 Mean: {post_mean:.3f}")
            print(f"  Change: {diff:+.3f} ({pct_change:+.1f}%)")
            print(f"  Statistically Significant: {'Yes' if significant else 'No'} (p={p_value:.4f})")
            
            if significant:
                if diff > 0:
                    print(f"  Interpretation: Significant INCREASE in {param} levels after 1995")
                else:
                    print(f"  Interpretation: Significant DECREASE in {param} levels after 1995")
            else:
                print(f"  Interpretation: No significant change in {param} levels")

def main():
    """Main function to run the detailed analysis"""
    
    print("Loading and cleaning water quality data...")
    df = load_and_clean_data()
    
    # Create subsets for pre-1995 and post-1995
    pre_1995 = df[df['Dates'] < '1995-01-01']
    post_1995 = df[df['Dates'] >= '1995-01-01']
    
    # Define the columns to analyze
    columns_mapping = {
        'E. coli': 'E. coli (cfu/100mL)',
        'BOD5': '5-day Biochemical Oxygen Demand (mg/L)',
        'NH3-N': 'Ammonia Nitrogen (mg/L)',
        'TIN': 'Total Inorganic Nitrogen (mg/L)',
        'TP': 'Total Phosphorus (mg/L)',
        'SS': 'Suspended Solids (mg/L)'
    }
    
    # Calculate comparison statistics
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
    
    comparison_df = pd.DataFrame(results).T
    
    # Perform statistical tests
    print("Performing statistical tests...")
    stats_results = perform_statistical_tests(pre_1995, post_1995, columns_mapping)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(pre_1995, post_1995, columns_mapping)
    
    # Print detailed analysis
    print_detailed_analysis(comparison_df, stats_results)
    
    # Save detailed results
    comparison_df.to_csv('detailed_water_quality_comparison.csv')
    pd.DataFrame(stats_results).T.to_csv('statistical_test_results.csv')
    
    print(f"\nDetailed results saved to:")
    print(f"  - detailed_water_quality_comparison.csv")
    print(f"  - statistical_test_results.csv")
    print(f"  - water_quality_comparison_plots.png")
    
    return comparison_df, stats_results

if __name__ == "__main__":
    main() 