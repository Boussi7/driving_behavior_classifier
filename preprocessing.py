import pandas as pd

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

# Creating the new CSV files and dataframes for cleaned data
normal_driving_df = process_driving_data('normal_data.csv', 'clean_normal_data.csv')
testing_data_df = process_driving_data('driving_data.csv', 'clean_driving_data.csv')