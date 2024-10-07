import math
import pandas as pd
from pprint import pprint
import re
import wandb

api = wandb.Api()

project = 'ai2-llm/olmo-medium'
group = "peteish7"
# name = "mitchish1"

runs_raw = api.runs(project)
runs = []

for run in runs_raw:
    if run.group == group:
    # if run.name == name:
        print()
        print(f"global_train_batch_size: {run.config['global_train_batch_size']}")
        print(f"device_train_microbatch_size: {run.config['device_train_microbatch_size']}")
        print(f"device_train_grad_accum: {run.config['device_train_grad_accum']}")
        # if run.config['device_train_grad_accum'] == 0:
            # continue
        # else:
        num_gpus = int(run.config['global_train_batch_size'] / (run.config['device_train_microbatch_size'] * run.config['device_train_grad_accum']))
        runs.append((num_gpus, run))
        print(f"Group: {run.group}")
        print(f"Name: {run.name}")

pprint(len(runs))
quit()


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