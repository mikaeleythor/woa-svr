# Preprocessor Z
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'C:\\Users\\Gunnar Ólafsson\\OneDrive - Reykjavik University\\Desktop\\Location4.csv'
df = pd.read_csv(file_path)

# Columns to normalize (excluding 'Power', which is already normalized)
columns_to_normalize = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
                        'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
                        'winddirection_100m', 'windgusts_10m']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply Z-Normalization (Standardization) to the selected columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Save the standardized dataframe to a new CSV file
df.to_csv(r'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_normalized_z_norm.csv', index=False)

print("Z-Normalization complete. The standardized file is saved as 'Location4_standardized.csv'.")
