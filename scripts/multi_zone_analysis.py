import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

# Constants
WQ_DATA_PATH = '../EPD_data/1_marine_water_quality.csv'
STATION_LOC_PATH = '../EPD_data/station_locations.csv'
OUTPUT_DIR = '../reports/figures'

KEY_PARAMETERS = [
    '5-day Biochemical Oxygen Demand (mg/L)', 'Suspended Solids (mg/L)',
    'Total Phosphorus (mg/L)', 'Total Nitrogen (mg/L)',
    'Ammonia Nitrogen (mg/L)', 'Orthophosphate Phosphorus (mg/L)',
    'Total Inorganic Nitrogen (mg/L)', 'Chlorophyll-a (μg/L)',
    'Dissolved Oxygen (%saturation)', 'Dissolved Oxygen (mg/L)',
    'E. coli (cfu/100mL)', 'Turbidity (NTU)', 'pH', 'Temperature (°C)',
    'Salinity (psu)'
]

def load_and_merge_data(wq_path, loc_path):
    """Loads water quality and station location data, then merges them."""
    try:
        df_wq = pd.read_csv(wq_path)
        df_loc = pd.read_csv(loc_path)
        print("Successfully loaded both CSV files.")
    except FileNotFoundError:
        print("Error: One or both CSV files not found.")
        return None

    # Perform a left merge to keep all water quality records
    merged_df = pd.merge(df_wq, df_loc, on='Station', how='left')

    # Report any water quality stations missing location data
    missing_location = merged_df[merged_df['Lat'].isnull()]['Station'].unique()
    if missing_location.size > 0:
        print("Stations in df_wq missing location data in df_loc:")
        print(missing_location)
    else:
        print("No stations in df_wq are missing location data in df_loc.")

    return merged_df

def clean_data(df, key_params):
    """Cleans the merged dataframe."""
    # Convert 'Dates' column to datetime and handle errors
    df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
    df.dropna(subset=['Dates'], inplace=True)
    df.set_index('Dates', inplace=True)

    # Convert key parameter columns to numeric
    for col in key_params:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\nNaN counts per key parameter column after cleaning:")
    print(df[key_params].isnull().sum())

    return df

def plot_station_map(df, output_dir):
    """Creates and saves a map of unique station locations."""
    if 'Lon' not in df.columns or 'Lat' not in df.columns:
        print("Longitude/Latitude columns not found. Skipping map plot.")
        return

    unique_stations = df[['Lon', 'Lat', 'Station']].drop_duplicates()
    gdf = gpd.GeoDataFrame(
        unique_stations,
        geometry=gpd.points_from_xy(unique_stations.Lon, unique_stations.Lat),
        crs="EPSG:4326"
    )

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([113.8, 114.5, 22.1, 22.6], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    ax.scatter(gdf.Lon, gdf.Lat, transform=ccrs.PlateCarree(),
               c='red', s=50, edgecolor='black', zorder=5)

    for x, y, label in zip(gdf.Lon, gdf.Lat, gdf.Station):
        ax.text(x + 0.01, y, label, transform=ccrs.PlateCarree(), fontsize=9)

    ax.set_title('Station Locations')
    plt.savefig(os.path.join(output_dir, 'station_locations_map.png'), dpi=300)
    print(f"\nStation map saved to {os.path.join(output_dir, 'station_locations_map.png')}")
    plt.close()

def plot_monthly_time_series(df, key_params, output_dir):
    """Calculates monthly averages and plots time series for each parameter."""
    monthly_averages = df[key_params].resample('ME').mean()

    # Create a multi-panel plot
    plt.style.use('seaborn-v0_8-whitegrid')
    num_params = len(key_params)
    num_cols = 3
    num_rows = int(np.ceil(num_params / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
    axes = axes.flatten()

    for i, param in enumerate(key_params):
        monthly_averages[param].plot(ax=axes[i])
        axes[i].set_title(param, fontsize=10)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Monthly Average')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Monthly Averages of Water Quality Parameters', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_averages_timeseries.png'), dpi=300)
    print(f"Time series plot saved to {os.path.join(output_dir, 'monthly_averages_timeseries.png')}")
    plt.close()

def main():
    """Main function to run the analysis."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load and merge data
    merged_df = load_and_merge_data(WQ_DATA_PATH, STATION_LOC_PATH)
    if merged_df is None:
        return

    # Clean data
    cleaned_df = clean_data(merged_df.copy(), KEY_PARAMETERS)

    # Display descriptive stats
    print("\nDescriptive Statistics for Key Parameters:")
    print(cleaned_df[KEY_PARAMETERS].describe())

    # Generate plots
    plot_station_map(cleaned_df, OUTPUT_DIR)
    plot_monthly_time_series(cleaned_df, KEY_PARAMETERS, OUTPUT_DIR)

if __name__ == '__main__':
    main()

