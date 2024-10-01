import math
import pandas as pd
from pprint import pprint
import re
import wandb
import json

api = wandb.Api()

project = 'ai2-llm/olmo-medium'
# group = "amberish7"
# name = "mitchish1"
groups = set()

runs_raw = api.runs(project)
runs = []

for run in runs_raw:
    # if run.group == group:
    # if run.name == name:
        if "global_train_batch_size" not in run.config or \
                "global_train_batch_size" not in run.config or \
                    "device_train_grad_accum" not in run.config or \
                        run.config['device_train_grad_accum'] == 0:
            continue
        print()
        print(f"global_train_batch_size: {run.config['global_train_batch_size']}")
        print(f"device_train_microbatch_size: {run.config['device_train_microbatch_size']}")
        print(f"device_train_grad_accum: {run.config['device_train_grad_accum']}")
        # else:
        num_gpus = int(run.config['global_train_batch_size'] / (run.config['device_train_microbatch_size'] * run.config['device_train_grad_accum']))
        try:
            meta = json.load(run.file("wandb-metadata.json").download(replace=True))
        except:
            continue
        if "gpu_devices" in meta and len(meta["gpu_devices"]) > 0 and "name" in meta["gpu_devices"][0] and "H100" in meta["gpu_devices"][0]["name"]:
            runs.append((num_gpus, run))
            groups.add(run.group)
            print(f"Group: {run.group}")
            print(f"Name: {run.name}")

print()
print(groups)
# quit()

# pprint(runs)
# quit()

kwh = 0.
gpu_hours = 0.

key_regex = re.compile(r'system\.gpu\..\.powerWatts')

all_keys = set()
for (num_gpus, run) in runs:
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

# Coreweave, on the east coast
print()
print(f'Total gpu hours: {gpu_hours / 3600}')
print(f'Total kwh: {kwh}')

# print(all_keys)