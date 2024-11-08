#Seperator_to_night_and_day
import pandas as pd

# Load the dataset from your local file
file_path = r'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_normalized_z_norm.csv'
df = pd.read_csv(file_path)

# Convert "Time" column to datetime, assume day-first format
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)

# Function to split day and night based on time
def get_day_night(hour):
    if 6 <= hour < 18:  # Define 6 AM to 6 PM as daytime
        return 'day'
    else:
        return 'night'

# Apply the day/night function to the data
df['Day_Night'] = df['Time'].apply(lambda x: get_day_night(x.hour))

# Filter the data to start from the first valid full day-night cycle
start_index = df[df['Day_Night'] == 'day'].index[0] if df[df['Day_Night'] == 'day'].index[0] < df[df['Day_Night'] == 'night'].index[0] else df[df['Day_Night'] == 'night'].index[0]
df_valid = df.loc[start_index:]

# Define the number of weeks to extract (up to 4)
num_weeks = 4
week_duration = pd.Timedelta(days=7)

# Process data week by week
for week_num in range(1, num_weeks + 1):
    # Calculate the start and end dates for the current week
    start_date = df_valid['Time'].min() + (week_num - 1) * week_duration
    end_date = start_date + week_duration
    
    # Extract data for the current week
    df_week = df_valid[(df_valid['Time'] >= start_date) & (df_valid['Time'] < end_date)]
    
    # If no data is available for the current week, stop processing
    if df_week.empty:
        print(f"No data available for week {week_num}. Stopping.")
        break
    
    # Split the data into daytime and nighttime
    day_data = df_week[df_week['Day_Night'] == 'day']
    night_data = df_week[df_week['Day_Night'] == 'night']
    
    # Save daytime and nighttime data for the current week
    day_file_path = fr'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_daytime_week{week_num}.csv'
    night_file_path = fr'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_nighttime_week{week_num}.csv'
    
    day_data.to_csv(day_file_path, index=False)
    night_data.to_csv(night_file_path, index=False)
    
    print(f"Daytime and nighttime data for week {week_num} have been saved successfully.")

print("Processing completed.")
