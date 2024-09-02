import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Compress time to ensure no gaps larger than max_diff
def compress_time(df, max_diff=2):
    for i in range(1, len(df)):
        time_diff = df.loc[i, 'SECONDS'] - df.loc[i-1, 'SECONDS']
        if time_diff > max_diff:
            df.loc[i, 'SECONDS'] = df.loc[i-1, 'SECONDS'] + max_diff
    
    return df

def process_driving_data(input_csv_file, output_csv_file):
    # Load the CSV file
    df = pd.read_csv(input_csv_file)

    # Filter the dataframe to keep only specified parameters
    parameters_to_keep = ["Vehicle speed", "Instant engine power (based on fuel consumption)", "Vehicle acceleration", "Engine RPM"]
    df = df[df['PID'].isin(parameters_to_keep)]

    # Remove moments where the engine is idle or stopped
    df = df[~((df['PID'] == "Vehicle acceleration") & (df['VALUE'] == 0))]
    df = df[~((df['PID'] == "Engine RPM") & (df['VALUE'] < 800))]
    df = df[~((df['PID'] == "Vehicle speed") & (df['VALUE'] == 0))]
    df = df[~((df['PID'] == "Instant engine power (based on fuel consumption)") & (df['VALUE'] < 100))]

    # Reset index to ensure sequential indexing
    df.reset_index(drop=True, inplace=True)

    # Reset SECONDS to start at 0 and increment by the change from row to row
    start_seconds = df['SECONDS'].iloc[0]
    df['SECONDS'] = df['SECONDS'] - start_seconds

    compressed_df = compress_time(df)

    # Save cleaned data to a new CSV
    compressed_df.to_csv(output_csv_file, index=False)

    return compressed_df

# Accept user inupt for file
def is_valid_filename(filename):
    # Check if the filename is not empty
    if not filename:
        print("Filename cannot be empty.")
        return False

    # Check for invalid characters in the filename
    invalid_chars = '<>:"/\\|?*'

    for char in invalid_chars:
        if char in filename:
            print(f"Filename contains an invalid character: {char}")
            return False
    
    # Make sure the filename ends in a csv
    if not filename.endswith(".csv"):
        print("Filename must end with '.csv'.")
        return False
    
    return True

class FileEmptyError(Exception):
    """Exception raised when a file is empty."""
    pass

def open_file_safely(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            if not content:
                raise FileEmptyError(f"The file '{filepath}' is empty.")
        return True
            
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and try again.")
    except IOError:
        print("Error: Unable to open the file. Please check the file and try again.")
    return False  # File did not open

# Ask user for file path
filepath = input("Please enter the path to a csv file with driving data: ")

# Creating the new CSV files and dataframes for cleaned data
normal_data = process_driving_data('normal_data.csv', 'clean_normal_data.csv')

# Validates the user-entered file
if open_file_safely(filepath):
    driving_data = process_driving_data(filepath, 'clean_driving_data.csv')
else:
    exit(1)

# Creates a column representing what interval a specific row of data is
# Intervals are 3s long
# [0,3)s is interval 1, [3,6)s is interval 2, and so on
normal_data['interval'] = (normal_data['SECONDS'] // 3).astype(int)
driving_data['interval'] = (driving_data['SECONDS'] // 3).astype(int)

# Makes each PID a column, makes calculations easier down the line
pivoted_data = normal_data.pivot_table(index='interval', columns='PID', values='VALUE', aggfunc='mean')
driving_pivoted_data = driving_data.pivot_table(index='interval', columns='PID', values='VALUE', aggfunc='mean')

# Calculate Z-Scores for each PID
z_scores = pivoted_data.apply(lambda x: (x - x.mean()) / x.std())
driving_z_scores = driving_pivoted_data.apply(lambda x: (x - x.mean()) / x.std())

# Apply weights (some PID's matter more than others)
# CHECK: This is subjective to my interpretation -- might require some more manual analysis of data/thought

# Speed is 0.55 because it can fluctuate heavily off a stoplight--do not want to misrepresent what aggressive driving is
# Acceleration is 0.7 because heavy fluctuation in acceleration represents aggressive speeding/braking

# Engine RPM can be misrepresented by a gear shift, and the horsepower metric is based on fuel consumption which is not necessarily
# An accurate representation of aggressive driving
weights = {
    'Vehicle speed': 0.55,
    'Vehicle acceleration': 0.7,
    'Engine RPM': 0.4,
    'Horsepower': 0.2
}

weighted_z_scores = z_scores.copy()
driving_weighted_z_scores = driving_z_scores.copy()

for col in weights.keys():
    if col in weighted_z_scores.columns:
        weighted_z_scores[col] = weighted_z_scores[col] * weights[col]

    if col in driving_weighted_z_scores.columns:
        driving_weighted_z_scores[col] = driving_weighted_z_scores[col] * weights[col]

# Sum the weighted Z-Scores to get a final score
# This is the metric that we will "judge/classify" other 3s samples with in the future
pivoted_data['final_score'] = weighted_z_scores.sum(axis=1)
driving_pivoted_data['final_score'] = driving_weighted_z_scores.sum(axis=1)

# Calculate mean and standard deviation of the final scores
# ONLY for the NORMAL DRIVING DATA
mean_score = pivoted_data['final_score'].mean()
std_dev_score = pivoted_data['final_score'].std()

# Classify the scores
# .where() works like a ternary operator
pivoted_data['classification'] = np.where(
    pivoted_data['final_score'] > mean_score + std_dev_score, 'Aggressive',
    np.where(pivoted_data['final_score'] < mean_score - std_dev_score, 'Slow', 'Normal')
)

# Plot the final scores over time to visualize data
# Helps because most of the "NORMAL" data should fall under the normal range (+/- 1 std dev from the mean of this dataset)
plt.figure(figsize=(14, 7))
plt.plot(pivoted_data.index * 3, pivoted_data['final_score'], label='Driving Score', color='blue')
plt.axhline(mean_score, color='green', linestyle='--', label='Mean')
plt.axhline(mean_score + std_dev_score, color='red', linestyle='--', label='+1 Std Dev (Aggressive)')
plt.axhline(mean_score - std_dev_score, color='orange', linestyle='--', label='-1 Std Dev (Slow)')
plt.xlabel('Time (s)')
plt.ylabel('Driving Score')
plt.title('Driving Behavior Over Time')
plt.legend()
plt.show()

## FROM HERE: Classifying the driving data that isn't 'normal', contains more data and should help the model be more accurate
## The purpose of having the "normal" data was to find a "line" to classify samples

driving_pivoted_data['classification'] = np.where(
    driving_pivoted_data['final_score'] > mean_score + std_dev_score, 'Aggressive',
    np.where(driving_pivoted_data['final_score'] < mean_score - std_dev_score, 'Slow', 'Normal')
)

# Plot the final scores over time to visualize data (for the driving dataset)
plt.figure(figsize=(14, 7))
plt.plot(driving_pivoted_data.index * 3, driving_pivoted_data['final_score'], label='Driving Score', color='blue')
plt.axhline(mean_score, color='green', linestyle='--', label='Mean')
plt.axhline(mean_score + std_dev_score, color='red', linestyle='--', label='+1 Std Dev (Aggressive)')
plt.axhline(mean_score - std_dev_score, color='orange', linestyle='--', label='-1 Std Dev (Slow)')
plt.xlabel('Time (s)')
plt.ylabel('Driving Score')
plt.title('Driving Behavior Over Time')
plt.legend()
plt.show()

# Map the classification back to driving_data & save as a new CSV for model training
driving_data['classification'] = driving_data['interval'].map(driving_pivoted_data['classification'])
driving_data.to_csv('classified_driving_data.csv', index=False)
