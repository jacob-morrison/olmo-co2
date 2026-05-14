import math
import pandas as pd
from pprint import pprint
import re
import wandb
from collections import Counter
import time

api = wandb.Api(timeout=300)  # Increase timeout

project = 'eai-ai2/2025_10_02_phase2'
name = "phase2.0_large_lr0.0001_wd0.002"

runs_raw = api.runs(project)
runs = []
cluster_counts = Counter()
for run in runs_raw:
    if run.name == name:
        num_gpus = run.metadata['gpu_count']
        runs.append((num_gpus, run))
        print(f"Group: {run.group}")
        print(f"Name: {run.name}")

print(cluster_counts)

kwh = 0.
gpu_hours = 0.

key_regex = re.compile(r'system\.gpu\..\.powerWatts')
sequential_data = []

all_keys = set()

def fetch_history_with_retry(run, max_retries=3, samples=1000):
    """Fetch run history with retry logic and sampling"""
    for attempt in range(max_retries):
        try:
            print(f"Fetching history for {run.name} (attempt {attempt + 1}/{max_retries})")
            # Use samples parameter to limit data size
            history = run.history(stream="events", samples=samples)
            return history
        except (wandb.errors.CommError, Exception) as e:
            print(f"Error fetching history: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch history after {max_retries} attempts. Skipping run.")
                return None

for (num_gpus, run) in runs:
    power_keys_list = []
    
    # Fetch history with retry logic
    history = fetch_history_with_retry(run, max_retries=3, samples=50000)
    if history is None:
        print(f"Skipping run {run.name} due to fetch errors")
        continue
    
    try:
        for i, row in history.iterrows():
            power_keys = {}
            for key in row.keys():
                all_keys.add(key)
                if key_regex.search(key) and not math.isnan(row[key]):
                    power_keys[key] = row[key]
                    power_keys['_timestamp'] = row['_timestamp']
            if len(power_keys) > 0:
                power_keys_list.append(power_keys)
                sequential_data.append({
                    'timestamp': power_keys['_timestamp'],
                    'GPU 0': power_keys.get('system.gpu.0.powerWatts', float('nan')),
                    'GPU 1': power_keys.get('system.gpu.1.powerWatts', float('nan')),
                    'GPU 2': power_keys.get('system.gpu.2.powerWatts', float('nan')),
                    'GPU 3': power_keys.get('system.gpu.3.powerWatts', float('nan')),
                    'GPU 4': power_keys.get('system.gpu.4.powerWatts', float('nan')),
                    'GPU 5': power_keys.get('system.gpu.5.powerWatts', float('nan')),
                    'GPU 6': power_keys.get('system.gpu.6.powerWatts', float('nan')),
                    'GPU 7': power_keys.get('system.gpu.7.powerWatts', float('nan')),
                })
    except Exception as e:
        print(f"Error processing history for run {run.name}: {e}")
        continue

    if len(power_keys_list) == 0:
        continue
        
    start_time = power_keys_list[-1]['_timestamp']
    end_time = power_keys_list[0]['_timestamp']
    total_time = start_time - end_time
    
    gpu_hours += total_time * num_gpus

    wattage = {}
    for i in range(1, len(power_keys_list)):
        for key in power_keys_list[i]:
            if key_regex.search(key):
                weighted_watts = power_keys_list[i][key] * (power_keys_list[i]['_timestamp'] - power_keys_list[i-1]['_timestamp'])
                if key not in wattage:
                    wattage[key] = 0.
                wattage[key] += weighted_watts

    print()

    total_watts = 0.
    for key in wattage:
        total_watts += wattage[key]

    print(f'Total time in hours for this job: {total_time / 3600}')
    print(f'Total kw seconds: {total_watts / 1000}')
    total_kwh = (total_watts / 1000) / 3600
    print(f'Total kwh: {total_kwh}')

    kwh += total_kwh

print()
print(f'Total gpu hours: {gpu_hours / 3600}')
print(f'Total kwh: {kwh}')
df = pd.DataFrame.from_dict(sequential_data)
print(df)
df.to_csv("dataframes/1b-power.csv")