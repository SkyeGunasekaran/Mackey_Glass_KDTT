import pandas as pd

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('Dataset/1.csv')

# Assuming your columns are named 'Time', 'Value', 'Anomaly', 'Dummy'
# Iterate through the rows to identify the first occurrence in consecutive anomalies
n_timesteps = 1000  # You can adjust this based on your preference

pre_anomaly_flag = False

for index, row in df.iterrows():
    if row['is_anomaly'] == 1 and not pre_anomaly_flag:
        # Label the preceding n timesteps as "pre-anomaly"
        for i in range(1, n_timesteps + 1):
            if index - i >= 0:
                df.at[index - i, 'is_anomaly'] = 2  # 2 denotes "pre-anomaly"
        pre_anomaly_flag = True
    elif row['is_anomaly'] == 0:
        pre_anomaly_flag = False

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_data.csv', index=False)
