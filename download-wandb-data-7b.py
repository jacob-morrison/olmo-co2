import math
import pandas as pd
from pprint import pprint
import re
import wandb

api = wandb.Api()

project = 'ai2-llm/olmo-medium'
lumi_key = 'v1_5-mix-mitch-ish-lumi'
mosaicml_key = 'v1_5-mix-mitch-ish-mcli'

runs = api.runs('ai2-llm/olmo-medium')

lumi_runs = []
mosaicml_runs = []

for run in runs:
    if lumi_key == run.name and '-2x' not in run.name: # TODO: see if we use *-2x runs or not
        lumi_runs.append(run)
    if mosaicml_key in run.name:
        mosaicml_runs.append(run)

print(len(lumi_runs))
print(len(mosaicml_runs))

# test_run = lumi_runs[0]

lumi_kwh = 0.
lumi_gpu_hours = 0.
mosaicml_kwh = 0.
mosaicml_gpu_hours = 0.

key_regex = re.compile(r'system\.gpu\..\.powerWatts')

all_keys = set()
for run in lumi_runs + mosaicml_runs:
# for run in [mosaicml_runs[0]]:
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
    
    if lumi_key in run.name:
        lumi_gpu_hours += total_time
    elif mosaicml_key in run.name:
        mosaicml_gpu_hours += total_time

    wattage = {}
    for i in range(1, len(power_keys_list)):
        for key in power_keys_list[i]:
            if key_regex.search(key):
                weighted_watts = power_keys_list[i][key] * (power_keys_list[i]['_timestamp'] - power_keys_list[i-1]['_timestamp'])
                if key not in wattage:
                    wattage[key] = 0.
                wattage[key] += weighted_watts

    print(wattage)

    total_watts = 0.
    for key in wattage:
        total_watts += wattage[key]

    print(f'Total time in hours for this job: {total_time / 3600}')
    print(f'Total kw seconds: {total_watts / 1000}')
    total_kwh = (total_watts / 1000) / 3600
    print(f'Total kwh: {total_kwh}')

    if lumi_key in run.name:
        lumi_kwh += total_kwh
    elif mosaicml_key in run.name:
        mosaicml_kwh += total_kwh
    else:
        print(f'Incorrect name? {run.name}')

# TODO: hardcoded multipliers are from Pete, update if we use *-lumi-2x runs
print()
print(f'Total gpu hours for lumi: {128 * 4 * lumi_gpu_hours / 3600}')
print(f'Total kwh for lumi: {128 * lumi_kwh}')
print(f'Total gpu hours for mosaic: {27 * 8 * mosaicml_gpu_hours / 3600}')
print(f'Total kwh for mosaicml: {27 * mosaicml_kwh}')

print(all_keys)