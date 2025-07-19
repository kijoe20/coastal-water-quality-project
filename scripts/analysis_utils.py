"""
Analysis utility functions for CIVL7009 water quality data analysis.

This module contains reusable functions for statistical analysis including
Mann-Kendall trend tests and Sen's slope calculations.
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.special import erf


def mann_kendall_test(x):
    """Performs the Mann-Kendall trend test.
    
    Args:
        x (array-like): Time series data
        
    Returns:
        float: p-value of the Mann-Kendall test
    """
    n = len(x)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p_value = 2 * (1 - abs(0.5 * (1 + erf(z / np.sqrt(2)))))
    return p_value


def sen_slope(x):
    """Calculate Sen's slope.
    
    Args:
        x (array-like): Time series data
        
    Returns:
        float: Sen's slope (median of all possible slopes)
    """
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    return np.median(slopes)