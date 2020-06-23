import numpy as np
from scipy.interpolate import interp1d

# Take a non-unique curve and return its unique values
def return_auc_function(fpr, tpr):
    # Find all unique fpr values
    unique_fpr = np.flip(np.unique(fpr))
    # Output
    matched_tpr = np.zeros((len(unique_fpr)))
    
    # For each value, pick the max corresponding tpr
    for idx, unique_value in enumerate(unique_fpr):
        # Where does equality occur
        matched_tpr[idx] = np.max(tpr[fpr == unique_value])
        
    return unique_fpr, matched_tpr

# Combine lists of curves (in fpr, tpr) using vertical averaging
def vertical_average(fpr_list, tpr_list, fpr_range):
    # Averaged TPR
    average_tpr = np.zeros((len(fpr_range)))
    num_points  = len(fpr_range)
    
    # For each point, interpolate all curves
    for point_idx in range(num_points):
        # Get the interpolated TPR
        average_tpr[point_idx] = np.mean([interp1d(fpr_list[curve_idx],
                                        tpr_list[curve_idx])(fpr_range[point_idx]) for
    curve_idx in range(len(fpr_list))])
    
    return average_tpr