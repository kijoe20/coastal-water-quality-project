# Multi-Zone Water Quality Analysis

This project provides a comprehensive analysis of water quality data across 10 different zones in Hong Kong's coastal waters. The analysis includes trend analysis using Mann-Kendall tests, correlation analysis, and visualization of time series data.

## Overview

The analysis processes 10 CSV files containing marine water quality data from different zones:

- Zone 1: Tolo Harbour and Channel
- Zone 2: Southern
- Zone 3: Port Shelter
- Zone 4: Junk Bay
- Zone 5: Deep Bay
- Zone 6: North Western
- Zone 7: Mirs Bay
- Zone 8: Western Buffer
- Zone 9: Eastern Buffer
- Zone 10: Victoria Harbour

## Features

### 1. Time Series Analysis

- Generates time series plots for all 15 water quality parameters in each zone
- Focuses on surface water data for consistency
- Handles missing values and non-numeric data appropriately

### 2. Trend Analysis (Mann-Kendall Test)

- Calculates Mann-Kendall p-values for trend significance
- Computes Sen's slope for trend magnitude and direction
- Creates master table with results for all zones and parameters

### 3. Correlation Analysis

- Generates correlation matrices for each zone
- Creates correlation heatmaps for visual interpretation
- Produces master table of correlation coefficients between parameters

### 4. Summary Statistics

- Identifies significant trends (p < 0.05) across all zones
- Highlights strong correlations (|r| > 0.7)
- Creates summary visualizations for easy comparison

## Water Quality Parameters Analyzed

1. 5-day Biochemical Oxygen Demand (mg/L)
2. Suspended Solids (mg/L)
3. Total Phosphorus (mg/L)
4. Total Nitrogen (mg/L)
5. Ammonia Nitrogen (mg/L)
6. Orthophosphate Phosphorus (mg/L)
7. Total Inorganic Nitrogen (mg/L)
8. Chlorophyll-a (μg/L)
9. Dissolved Oxygen (%saturation)
10. Dissolved Oxygen (mg/L)
11. E. coli (cfu/100mL)
12. Turbidity (NTU)
13. pH
14. Temperature (°C)
15. Salinity (psu)

## Installation and Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Data Structure**
   Ensure your `EPD_data/` directory contains the 10 CSV files:
   - `1_marine_water_quality.csv`
   - `2_marine_water_quality.csv`
   - `3_marine_water_quality.csv`
   - `4_marine_water_quality.csv`
   - `5_marine_water_quality.csv`
   - `6_marine_water_quality.csv`
   - `7_marine_water_quality.csv`
   - `8_marine_water_quality.csv`
   - `9_marine_water_quality.csv`
   - `10_marine_water_quality.csv`

## Usage

### Quick Start

Run the analysis from the project root directory:

```bash
python run_multi_zone_analysis.py
```

### Advanced Usage

For more control over the analysis, you can use the analyzer class directly:

```python
from scripts.multi_zone_analysis import MultiZoneWaterQualityAnalyzer

# Initialize analyzer with custom directories
analyzer = MultiZoneWaterQualityAnalyzer(
    data_dir="EPD_data",
    output_dir="my_results"
)

# Run the complete analysis
analyzer.run_analysis()
```

## Output Structure

After running the analysis, you'll find the following structure in the `multi_zone_results/` directory:

```
multi_zone_results/
├── plots/
│   ├── zone_1_[parameter]_timeseries.png
│   ├── zone_1_correlation_heatmap.png
│   ├── zone_2_[parameter]_timeseries.png
│   ├── zone_2_correlation_heatmap.png
│   └── ... (for all 10 zones)
│   ├── significant_trends_by_zone.png
│   └── average_correlation_by_zone.png
└── tables/
    ├── master_mk_test_results.csv
    ├── master_correlation_results.csv
    ├── significant_trends_summary.csv
    └── strong_correlations_summary.csv
```

## Output Files Explained

### Master Tables

1. **master_mk_test_results.csv**

   - Contains Mann-Kendall test results for all parameters across all zones
   - Columns: Zone, Parameter_mk_pvalue, Parameter_sen_slope, Parameter_n_samples
   - Use this to compare trend significance and magnitude between zones

2. **master_correlation_results.csv**

   - Contains correlation coefficients between all parameter pairs for each zone
   - Columns: Zone, Parameter1, Parameter2, Correlation
   - Use this to identify which parameters are strongly correlated in each zone

3. **significant_trends_summary.csv**

   - Lists only the parameters with statistically significant trends (p < 0.05)
   - Includes trend direction (Increasing/Decreasing)
   - Useful for identifying which zones show significant changes over time

4. **strong_correlations_summary.csv**
   - Lists only correlations with |r| > 0.7
   - Helps identify the strongest relationships between parameters

### Visualizations

1. **Time Series Plots**

   - One plot per parameter per zone
   - Shows temporal trends and variability
   - Named: `zone_[number]_[parameter]_timeseries.png`

2. **Correlation Heatmaps**

   - One heatmap per zone showing correlations between all parameters
   - Color-coded for easy interpretation
   - Named: `zone_[number]_correlation_heatmap.png`

3. **Summary Plots**
   - `significant_trends_by_zone.png`: Bar chart showing number of significant trends per zone
   - `average_correlation_by_zone.png`: Average correlation strength by zone

## Interpreting Results

### Mann-Kendall Test Results

- **P-value < 0.05**: Statistically significant trend
- **Sen's slope > 0**: Increasing trend
- **Sen's slope < 0**: Decreasing trend
- **Sen's slope magnitude**: Rate of change

### Correlation Results

- **|r| > 0.7**: Strong correlation
- **|r| 0.5-0.7**: Moderate correlation
- **|r| < 0.5**: Weak correlation
- **Positive r**: Parameters increase together
- **Negative r**: Parameters change in opposite directions

## Data Quality Notes

- Analysis focuses on surface water data for consistency
- Missing values and non-numeric entries are handled appropriately
- Minimum sample size requirements ensure statistical validity
- Results are filtered to exclude zones with insufficient data

## Troubleshooting

### Common Issues

1. **Import Errors**

   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're running from the project root directory

2. **Missing Data Files**

   - Verify that all 10 CSV files are present in the `EPD_data/` directory
   - Check file naming convention matches expected format

3. **Memory Issues**

   - The analysis processes large datasets; ensure sufficient RAM
   - Consider processing zones individually if memory is limited

4. **Plot Generation Errors**
   - Ensure write permissions in the output directory
   - Check that matplotlib backend is properly configured

## Customization

You can customize the analysis by modifying the `MultiZoneWaterQualityAnalyzer` class:

- Change significance threshold (default: p < 0.05)
- Modify correlation strength threshold (default: |r| > 0.7)
- Adjust minimum sample size requirements
- Add additional statistical tests
- Customize plot styling and output formats

## Contributing

To extend the analysis:

1. Add new statistical methods to `analysis_utils.py`
2. Modify the analyzer class to include new analyses
3. Update the output generation methods
4. Add new visualization types as needed

## License

This project is part of the CIVL7009 course work on coastal water quality analysis.
