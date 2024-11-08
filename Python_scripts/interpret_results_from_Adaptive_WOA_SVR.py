# interpet the results from the Adaptive WOA-SVR model.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import ast
import re  # New import for regular expression to handle whitespace

# Load results
file_path = r'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_Results.csv'
results_df = pd.read_csv(file_path)

# Extract week and time information from the 'File' column
results_df[['Time', 'Week']] = results_df['File'].str.extract(r'_(daytime|nighttime)_week(\d)')

# Enhanced function to clean up 'Best Params' and handle excess whitespace
def parse_best_params(param_string):
    # Remove multiple spaces or replace space-separated numbers with commas
    formatted_string = re.sub(r'\s+', ',', param_string.strip())
    try:
        # Safely evaluate using literal_eval
        return ast.literal_eval(formatted_string)
    except (SyntaxError, ValueError):
        return [None, None, None]  # Default values in case of parsing error

# Apply parsing function to the 'Best Params' column
results_df['Best Params'] = results_df['Best Params'].apply(parse_best_params)

# Drop rows with any None values in 'Best Params' after parsing
results_df = results_df.dropna(subset=['Best Params'])

# Find the best configurations for each time of day and week
best_results = results_df.loc[results_df.groupby(['Time', 'Week'])['Test RMSE'].idxmin()]

# Print the best configurations for each day/night and week
print("Best Results by Time of Day and Week:")
for _, row in best_results.iterrows():
    C, epsilon, gamma = row['Best Params']
    if C is not None and epsilon is not None and gamma is not None:
        print(f"\nTime: {row['Time']}, Week: {row['Week']}")
        print(f"Best Parameters: C={C}, epsilon={epsilon}, gamma={gamma}")
        print(f"Train RMSE: {row['Train RMSE']}, Test RMSE: {row['Test RMSE']}, Test R²: {row['Test R²']}")
    else:
        print(f"\nTime: {row['Time']}, Week: {row['Week']}")
        print("Best Parameters: Values missing for C, epsilon, or gamma")

# Brief statistical summary
print("\n--- Statistical Summary ---")

# Calculate mean and standard deviation for RMSE and R² by day/night
rmse_stats = results_df.groupby('Time')['Test RMSE'].agg(['mean', 'std'])
r2_stats = results_df.groupby('Time')['Test R²'].agg(['mean', 'std'])

print("\nAverage Test RMSE and R² by Time of Day:")
print(pd.concat([rmse_stats, r2_stats], axis=1, keys=['Test RMSE', 'Test R²']))

# Overall best results across all configurations
overall_best = results_df.loc[results_df['Test RMSE'].idxmin()]
C_best, epsilon_best, gamma_best = overall_best['Best Params']
print("\nOverall Best Configuration Across All Times and Weeks:")
print(f"Time: {overall_best['Time']}, Week: {overall_best['Week']}")
print(f"Best Parameters: C={C_best}, epsilon={epsilon_best}, gamma={gamma_best}")
print(f"Train RMSE: {overall_best['Train RMSE']}, Test RMSE: {overall_best['Test RMSE']}, Test R²: {overall_best['Test R²']}")

# Visualization of Test RMSE and R² by time of day and week
def plot_performance(day_df, night_df):
    plt.figure(figsize=(14, 6))

    # Plot Test RMSE over weeks
    plt.subplot(1, 2, 1)
    plt.plot(day_df['Week'], day_df['Test RMSE'], label='Daytime', marker='o', color='skyblue')
    plt.plot(night_df['Week'], night_df['Test RMSE'], label='Nighttime', marker='o', color='salmon')
    plt.xlabel("Week")
    plt.ylabel("Test RMSE")
    plt.title("Test RMSE by Week and Time of Day")
    plt.legend()

    # Plot Test R² over weeks
    plt.subplot(1, 2, 2)
    plt.plot(day_df['Week'], day_df['Test R²'], label='Daytime', marker='o', color='skyblue')
    plt.plot(night_df['Week'], night_df['Test R²'], label='Nighttime', marker='o', color='salmon')
    plt.xlabel("Week")
    plt.ylabel("Test R²")
    plt.title("Test R² by Week and Time of Day")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Separate DataFrames for daytime and nighttime
daytime_df = results_df[results_df['Time'] == 'daytime']
nighttime_df = results_df[results_df['Time'] == 'nighttime']

# Call the visualization function
plot_performance(daytime_df, nighttime_df)

# Statistical Analysis: t-test for difference in Test RMSE and Test R² between daytime and nighttime
rmse_t_stat, rmse_p_val = ttest_ind(daytime_df['Test RMSE'], nighttime_df['Test RMSE'], equal_var=False, nan_policy='omit')
r2_t_stat, r2_p_val = ttest_ind(daytime_df['Test R²'], nighttime_df['Test R²'], equal_var=False, nan_policy='omit')

print("\n--- T-Test Analysis ---")
print(f"T-test for Test RMSE between Daytime and Nighttime:")
print(f"t-statistic = {rmse_t_stat:.4f}, p-value = {rmse_p_val:.4f}")
print(f"T-test for Test R² between Daytime and Nighttime:")
print(f"t-statistic = {r2_t_stat:.4f}, p-value = {r2_p_val:.4f}")

# Interpretation based on p-values
alpha = 0.05
if rmse_p_val < alpha:
    print("\nSignificant difference in Test RMSE between daytime and nighttime.")
    print("Consider different approaches for daytime vs. nighttime data.")
else:
    print("\nNo significant difference in Test RMSE between daytime and nighttime.")

if r2_p_val < alpha:
    print("Significant difference in Test R² between daytime and nighttime.")
    print("Consider different approaches for daytime vs. nighttime data.")
else:
    print("No significant difference in Test R² between daytime and nighttime.")