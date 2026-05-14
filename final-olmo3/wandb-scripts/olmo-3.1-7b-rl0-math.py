import math
import pandas as pd
from pprint import pprint
import re
import wandb
from collections import Counter
from datetime import datetime, timezone

api = wandb.Api(timeout=60)

project = 'ai2-llm/open_instruct_internal'
# group = "OLMo25"
name1 = "olmo3_7b_rlzero_math__1__1763966683"
name2 = "olmo3_7b_rlzero_math_restart2k__1__1765400165"

projects = [
    'ai2-llm/open_instruct_internal'
]
# group = "OLMo25"
# name = "olmo3-7b-DPO"

def get_gpu_counts(run):
    config = run.config
    
    # Trainer GPUs
    # num_nodes = config["num_nodes"]
    num_learners_per_node = config["num_learners_per_node"]
    
    # Handle if it's a list or int
    if isinstance(num_learners_per_node, list):
        learners_per_node = int(sum(num_learners_per_node))
    else:
        learners_per_node = int(num_learners_per_node)
    
    trainer_gpus = learners_per_node

    # print(trainer_gpus)
    # assert(isinstance(trainer_gpus, int))
    
    # vLLM GPUs
    vllm_num_engines = config["vllm_num_engines"]
    vllm_tp_size = config.get("vllm_tensor_parallel_size", 1)
    vllm_gpus = vllm_num_engines * vllm_tp_size
    
    total_gpus = trainer_gpus + vllm_gpus
    
    return {
        "trainer_gpus": int(trainer_gpus),
        "vllm_gpus": int(vllm_gpus),
        "total_gpus": int(total_gpus),
        "vllm_num_engines": int(vllm_num_engines),
        "vllm_tensor_parallel_size": int(vllm_tp_size),
    }

def get_script_name(run):
    """Extract the script name from a run's rawconfig."""
    try:
        # Try code_path first (cleaner)
        code_path = run.rawconfig.get('_wandb', {}).get('code_path', '')
        if code_path:
            return code_path.split('/')[-1]  # Returns 'grpo_fast.py'
        
        # Fallback: check the environment entries
        wandb_config = run.rawconfig.get('_wandb', {}).get('e', {})
        for env_data in wandb_config.values():
            if 'codePath' in env_data:
                return env_data['codePath'].split('/')[-1]
            if 'program' in env_data:
                return env_data['program'].split('/')[-1]
    except (AttributeError, KeyError):
        pass
    return None

START_DATE = datetime(2025, 4, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 15, tzinfo=timezone.utc)

def is_in_date_range(run, start_date, end_date):
    """Check if run was created within the specified date range."""
    # run.created_at returns a datetime string like '2026-01-22T20:48:41Z'
    created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
    return start_date <= created_at <= end_date

for project in projects:
    runs_raw = api.runs(project)
    runs = []
    cluster_counts = Counter()
    for run in runs_raw:
        # if run.group == group:
        # if "grpo" in run.name.lower():
        if name1 not in run.name and name2 not in run.name:
            continue
        else:
            print("found it!")
            print(run.name)
            # print(run.settings)
            # print(vars(run))
            # quit()
            # print()
            # print(run.config)
            # clusters = run.config['launch']['clusters']
            # if len(clusters) > 1:
            #     print(clusters)
            # else:
            #     cluster_counts[clusters[0]] += 1
            # if "global_batch_size" not in run.config["data_loader"] or "rank_microbatch_size" not in run.config['train_module']:
            #     print(run.config)
            #     print()
            #     quit()
            # print(f"global_batch_size: {run.config['data_loader']['global_batch_size']}")
            # print(f"rank_microbatch_size: {run.config['train_module']['rank_microbatch_size']}")
            # if run.config['device_train_grad_accum'] == 0:
                # continue
            # else:
            # num_gpus = int(run.config['data_loader']['global_batch_size'] / (run.config['train_module']['rank_microbatch_size'])) * 4
            if "vllm_num_engines" not in run.config:
                print("vllm_num_engines")
                print("skipping run")
                continue
            # if "num_nodes" not in run.config:
                # print("num_nodes")
                # print("skipping run")
                # continue
            if "num_learners_per_node" not in run.config:
                print("num_learners_per_node")
                print("skipping run")
                continue
            # print("good run!!! #############################################################################")
            results = get_gpu_counts(run)
            trainer_gpus = results["trainer_gpus"]
            vllm_gpus = results["vllm_gpus"]
            if trainer_gpus < 8:
                continue
            # , total_gpus, vllm_num_engines, vllm_tensor_parallel_size
            # if "per_device_train_batch_size" not in run.config or "gradient_accumulation_steps" not in run.config:
                # print("skipping run")
                # continue
            # per_device_batch_size = run.config["per_device_train_batch_size"]
            # gradient_accumulation_steps = run.config["gradient_accumulation_steps"]

            # Your known effective batch size
            # effective_batch_size = 128

            # Calculate total GPUs
            # batch_per_gpu = per_device_batch_size * gradient_accumulation_steps
            # num_gpus = effective_batch_size / batch_per_gpu
            # num_gpus = 64
            runs.append((trainer_gpus, vllm_gpus, run))
            print(f"Group: {run.group}")
            print(f"Name: {run.name}")
            # break

print(cluster_counts)
# quit()
# pprint(len(runs))
# quit()

kwh = 0.
trainer_gpu_hours = 0.
vllm_gpu_hours = 0.

key_regex = re.compile(r'system\.gpu\..\.powerWatts')
sequential_data = []

all_keys = set()
for (trainer_gpus, vllm_gpus, run) in runs:
    power_keys_list = []
    try:
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
        
        # print()
        # print(type(total_time))
        # print(type(trainer_gpus))
        trainer_gpu_hours += total_time * trainer_gpus
        vllm_gpu_hours += total_time * vllm_gpus

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
    except:
        print("error")

    total_watts = 0.
    for key in wattage:
        total_watts += wattage[key]

    print(f'Total time in hours for this job: {total_time / 3600}')
    print(f'Total kw seconds: {total_watts / 1000}')
    total_kwh = (total_watts / 1000) / 3600
    print(f'Total kwh: {total_kwh}')

    kwh += total_kwh * (trainer_gpus / 8)

# Coreweave, on the east coast
print()
print(f'Total trainer gpu hours: {trainer_gpu_hours / 3600}')
print(f'Total trainer kwh: {kwh}')
print(f'Total vllm gpu hours: {vllm_gpu_hours / 3600}')
df = pd.DataFrame.from_dict(sequential_data)
print(df)
df.to_csv("dataframes/1b-power.csv")
# print(all_keys)