#Kl Convergence
import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Base directory for your data files
base_dir = r'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location1_skipt_vikur'

# List of filenames within the base directory
day_filenames = [
    "Location1_daytime_week1.csv",
    "Location1_daytime_week2.csv",
    "Location1_daytime_week3.csv",
    "Location1_daytime_week4.csv"
]

night_filenames = [
    "Location1_nighttime_week1.csv",
    "Location1_nighttime_week2.csv",
    "Location1_nighttime_week3.csv",
    "Location1_nighttime_week4.csv"
]

def load_and_combine_data(filenames):
    """Load and combine CSV files into a single DataFrame."""
    data_frames = []
    for filename in filenames:
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data['windspeed_10m'], combined_data['windspeed_100m']

def calculate_statistics(windspeed):
    """Calculate mean and variance of a windspeed column."""
    mean = np.mean(windspeed)
    variance = np.var(windspeed)
    return mean, variance

def kl_divergence(mean1, var1, mean2, var2):
    """Calculate KL divergence between two normal distributions."""
    return np.log(np.sqrt(var2) / np.sqrt(var1)) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5

# Load and combine daytime and nighttime windspeed data
day_windspeed_10m, day_windspeed_100m = load_and_combine_data(day_filenames)
night_windspeed_10m, night_windspeed_100m = load_and_combine_data(night_filenames)

# Calculate statistics for windspeed_10m and windspeed_100m
mean_day_10m, var_day_10m = calculate_statistics(day_windspeed_10m)
mean_night_10m, var_night_10m = calculate_statistics(night_windspeed_10m)
mean_day_100m, var_day_100m = calculate_statistics(day_windspeed_100m)
mean_night_100m, var_night_100m = calculate_statistics(night_windspeed_100m)

# Calculate KL divergence for windspeed_10m and windspeed_100m
kl_div_10m = kl_divergence(mean_day_10m, var_day_10m, mean_night_10m, var_night_10m)
kl_div_100m = kl_divergence(mean_day_100m, var_day_100m, mean_night_100m, var_night_100m)

# Print results
print(f"KL Divergence for windspeed_10m (day vs night): {kl_div_10m}")
print(f"KL Divergence for windspeed_100m (day vs night): {kl_div_100m}")

# Optional: Visualize distributions
plt.figure()
plt.hist(day_windspeed_10m, bins=30, alpha=0.5, label="Day Windspeed 10m", density=True)
plt.hist(night_windspeed_10m, bins=30, alpha=0.5, label="Night Windspeed 10m", density=True)
plt.title("Windspeed 10m Distribution (Day vs Night)")
plt.legend()

plt.figure()
plt.hist(day_windspeed_100m, bins=30, alpha=0.5, label="Day Windspeed 100m", density=True)
plt.hist(night_windspeed_100m, bins=30, alpha=0.5, label="Night Windspeed 100m", density=True)
plt.title("Windspeed 100m Distribution (Day vs Night)")
plt.legend()

plt.show()