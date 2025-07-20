#!/usr/bin/env python3
"""
Runner script for Multi-Zone Water Quality Analysis

This script runs the comprehensive analysis on all 10 water quality zones.
Simply run this script from the project root directory.

Usage:
    python run_multi_zone_analysis.py
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

try:
    from multi_zone_analysis import MultiZoneWaterQualityAnalyzer
    print("Multi-Zone Water Quality Analysis")
    print("=" * 50)
    print("This script will analyze all 10 water quality zones and generate:")
    print("1. Time series plots for each parameter in each zone")
    print("2. Master table of Mann-Kendall test results")
    print("3. Master table of correlation results")
    print("4. Summary visualizations")
    print("=" * 50)
    
    # Run the analysis
    analyzer = MultiZoneWaterQualityAnalyzer()
    analyzer.run_analysis()
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved in: {analyzer.output_dir}")
    print("\nGenerated files:")
    print(f"- Plots: {analyzer.output_dir}/plots/")
    print(f"- Tables: {analyzer.output_dir}/tables/")
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error running analysis: {e}")
    sys.exit(1) 