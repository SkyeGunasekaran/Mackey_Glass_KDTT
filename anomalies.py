import pandas as pd

def print_anomaly_info(df):
    anomaly_start = None
    anomaly_end = None
    time_between_anomalies = []

    for index, row in df.iterrows():
        if row['is_anomaly'] == 1:
            if anomaly_start is None:
                anomaly_start = row['time']
            anomaly_end = row['time']
        elif anomaly_start is not None:
            print(f"Anomaly at time {anomaly_start}, lasting until {anomaly_end}")
            anomaly_start = None

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('Dataset/1.csv')

# Assuming your columns are named 'Time', 'Value', 'Anomaly', 'Dummy'
# Call the function to print anomaly information
print_anomaly_info(df)
