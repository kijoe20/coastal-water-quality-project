# Water Quality PCA Analysis

This project performs Principal Component Analysis (PCA) on water quality data from multiple zones to identify patterns and trends in water quality parameters.

## Features

- **Automated Data Processing**: Loads and processes multiple CSV files from a specified directory
- **Data Preprocessing**: Handles missing values through imputation and standardizes data for PCA
- **Monthly Aggregation**: Resamples data to monthly averages for consistent temporal analysis
- **PCA Analysis**: Performs 2-component PCA to reduce dimensionality and identify major patterns
- **Visualization**: Creates time-series plots of PC1 with 12-month rolling means
- **Comprehensive Reporting**: Generates summary reports with key findings for all zones

## Directory Structure

```
pca_analysis/
├── pca_water_quality_analysis.py  # Main analysis script
├── requirements.txt               # Python dependencies
├── README.md                     # This file
└── outputs/                      # Generated outputs
    ├── pca_ready_data/          # Preprocessed data files
    ├── plots/                   # PC1 trend plots
    └── pca_summary_all_zones.csv # Final summary report
```

## Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Structure**: 
   - Place your water quality CSV files in `../EPD_data/` relative to this directory
   - Each CSV should have a 'Date' column and the following water quality parameters:
     - Temperature (°C)
     - Chlorophyll-a (μg/L)
     - Dissolved Oxygen (%saturation)
     - Salinity (psu)
     - pH
     - 5-day Biochemical Oxygen Demand (mg/L)
     - Suspended Solids (mg/L)
     - Total Nitrogen (mg/L)
     - Total Inorganic Nitrogen (mg/L)
     - Ammonia Nitrogen (mg/L)
     - Orthophosphate Phosphorus (mg/L)
     - Total Phosphorus (mg/L)

## Usage

Run the analysis script:

```bash
python pca_water_quality_analysis.py
```

## Output Files

1. **PCA-Ready Data** (`outputs/pca_ready_data/`):
   - Preprocessed and standardized data for each zone
   - Files named: `pca_ready_[zone_name].csv`

2. **Plots** (`outputs/plots/`):
   - PC1 time-series plots with rolling means
   - Files named: `pc1_trend_[zone_name].png`

3. **Summary Report** (`outputs/pca_summary_all_zones.csv`):
   - Comprehensive summary with:
     - Zone names
     - PC1 explained variance ratios
     - Top 3 positively correlated parameters for PC1
     - Top 3 negatively correlated parameters for PC1
     - Data quality metrics

## Analysis Workflow

1. **Data Loading**: Loads CSV files and parses dates
2. **Monthly Resampling**: Aggregates data to monthly averages
3. **Preprocessing**:
   - Imputes missing values using mean strategy
   - Standardizes features using StandardScaler
4. **PCA Execution**: Performs 2-component PCA analysis
5. **Visualization**: Creates PC1 trend plots with rolling means
6. **Results Aggregation**: Compiles findings into summary report

## Interpretation

- **PC1**: First principal component explaining the largest variance in the data
- **Loadings**: Show which parameters contribute most to each principal component
- **Positive correlations**: Parameters that increase together with PC1
- **Negative correlations**: Parameters that decrease when PC1 increases
- **Rolling mean**: Smoothed trend showing long-term patterns

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0

## Error Handling

The script includes robust error handling for:
- Missing or corrupted data files
- Insufficient data for analysis
- Missing feature columns
- File I/O operations

If errors occur, the script will skip problematic files and continue processing the remaining data.