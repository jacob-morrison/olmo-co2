import math
import pandas as pd
from pprint import pprint
import re
import wandb
from collections import Counter
from datetime import datetime, timezone

api = wandb.Api(timeout=60)

project = 'ai2-llm/olmo-cookbook'
# group = "OLMo25"
# name = "stego32-longcontext-run-3-20251107T0947+0000"

START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 15, tzinfo=timezone.utc)

def is_in_date_range(run, start_date, end_date):
    """Check if run was created within the specified date range."""
    # run.created_at returns a datetime string like '2026-01-22T20:48:41Z'
    created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
    return start_date <= created_at <= end_date


runs_raw = api.runs(project)
runs = []
cluster_counts = Counter()
for run in runs_raw:
    # if run.group == group:
    if is_in_date_range(run, START_DATE, END_DATE):
        print()
        # print(run.config)
        # clusters = run.config['launch']['clusters']
        # if len(clusters) > 1:
            # print(clusters)
        # else:
            # cluster_counts[clusters[0]] += 1
        if "data_loader" not in run.config or "train_module" not in run.config or "global_batch_size" not in run.config["data_loader"] or "rank_microbatch_size" not in run.config['train_module']:
            print(run.config)
            print()
            continue
        print(f"global_batch_size: {run.config['data_loader']['global_batch_size']}")
        print(f"rank_microbatch_size: {run.config['train_module']['rank_microbatch_size']}")
        # if run.config['device_train_grad_accum'] == 0:
            # continue
        # else:
        num_gpus = int(run.config['data_loader']['global_batch_size'] / (run.config['train_module']['rank_microbatch_size']))
        # num_gpus = 1024
        runs.append((num_gpus, run))
        print(num_gpus)
        print(f"Group: {run.group}")
        print(f"Name: {run.name}")

print(cluster_counts)
# quit()
# pprint(len(runs))
# quit()

kwh = 0.
gpu_hours = 0.

key_regex = re.compile(r'system\.gpu\..\.powerWatts')
sequential_data = []

all_keys = set()
for (num_gpus, run) in runs:
    try:
        power_keys_list = []
        for i, row in run.history(stream="events").iterrows():
            # print(run.name)
            power_keys = {}
            for key in row.keys():
                all_keys.add(key)
                if key_regex.search(key) and not math.isnan(row[key]):
                    power_keys[key] = row[key]
                    # print(row[key])
                    power_keys['_timestamp'] = row['_timestamp'] # in seconds?
            if len(power_keys) > 0:
                # print(power_keys['_timestamp'])
                power_keys_list.append(power_keys)
                if "system.gpu.1.powerWatts" not in power_keys:
                    continue
                sequential_data.append({
                    'timestamp': power_keys['_timestamp'],
                    'GPU 0': power_keys['system.gpu.0.powerWatts'],
                    'GPU 1': power_keys['system.gpu.1.powerWatts'],
                    'GPU 2': power_keys['system.gpu.2.powerWatts'],
                    'GPU 3': power_keys['system.gpu.3.powerWatts'],
                    'GPU 4': power_keys['system.gpu.4.powerWatts'],
                    'GPU 5': power_keys['system.gpu.5.powerWatts'],
                    'GPU 6': power_keys['system.gpu.6.powerWatts'],
                    'GPU 7': power_keys['system.gpu.7.powerWatts'],
                })

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
        # print(wattage)

        total_watts = 0.
        for key in wattage:
            total_watts += wattage[key]

        print(f'Total time in hours for this job: {total_time / 3600}')
        print(f'Total kw seconds: {total_watts / 1000}')
        total_kwh = (total_watts / 1000) / 3600
        print(f'Total kwh: {total_kwh}')

        kwh += total_kwh * (num_gpus / 8)
    except Exception as e:
        print(e)

# Coreweave, on the east coast
print()
print(f'Total gpu hours: {gpu_hours / 3600}')
print(f'Total kwh: {kwh}')
df = pd.DataFrame.from_dict(sequential_data)
print(df)
df.to_csv("dataframes/1b-power.csv")
# print(all_keys)