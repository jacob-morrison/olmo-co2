import math
import re
import wandb
import pandas as pd
from collections import defaultdict

api = wandb.Api(timeout=60)

# --- Individual run URLs ---
individual_run_urls = """
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/ci4kk6cf/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/sg3nke52/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/ilo6rxfy/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/8v18qwgd/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/oz13lw04/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/22fkmnjk
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/7iuy7k9t/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/8sqdb1qd/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/yl17npuy/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/bxxkah46/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/q4m37ql2/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/n2vi9o8c/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/3id4nl98/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/zgu3wzw0/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/jen0xpu2/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/624m9b04/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/mfhixmo8/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/0eaui6s9/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/vigsl0ts/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/qg1ivql7/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/7o0fze7s/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/5xtv5mp6/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/jixfw7o8/
https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/ll61anay
""".strip().split('\n')

# --- Group URLs ---
group_urls = """
https://wandb.ai/ai2-llm/olmo3/groups/OLMo-2.5-noswa
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.8-correctdata-noheadnorm
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B-gqa-16/workspace
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B-gqa-4/workspace
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B-nogqa/
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-float8?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-float8-same-init?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-gqa-fixed-max5T?nw=nwuseramandab
http://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-headnorm-test?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-plus-qknorm-reordered-norm-fixed-reload?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/OLMo-2.5-preorder-no-qk?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/OLMo-2.5-preorder
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-plus-qknorm?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-swa-fixed-init
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-swa
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-gqa
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-gqa-fixed?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-half-context-fixed-init?nw=nwuseramandab
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-halfcontext
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-halfcontext
https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B-4K
https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.9-2.5ish
https://wandb.ai/ai2-llm/olmo3/groups/OLMo-2.5-preorder
""".strip().split('\n')

# ---- Parsing helpers ----

def parse_run_url(url):
    m = re.search(r'wandb\.ai/([^/]+)/([^/]+)/runs/([^/?\s]+)', url.strip())
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return None

def parse_group_url(url):
    m = re.search(r'wandb\.ai/([^/]+)/([^/]+)/groups/([^/?\s]+)', url.strip())
    if m:
        return f"{m.group(1)}/{m.group(2)}", m.group(3)
    return None, None

# ---- GPU count helper ----

def get_num_gpus(run):
    """Determine num_gpus from config, trying multiple sources."""
    # 1. Try launch config (num_nodes * num_gpus_per_node)
    try:
        launch = run.config['launch']
        num_nodes = launch['num_nodes']
        gpus_per_node = launch.get('num_gpus', 8)
        return int(num_nodes) * int(gpus_per_node)
    except (KeyError, TypeError):
        pass

    # 2. Try global_batch_size / rank_microbatch_size
    try:
        gbs = run.config['data_loader']['global_batch_size']
        rms = run.config['train_module']['rank_microbatch_size']
        return int(gbs / rms)
    except (KeyError, TypeError):
        pass

    # 3. Fallback
    print(f"    WARNING: Could not determine num_gpus for {run.name}, defaulting to 16")
    return 16

# ---- Process a single run, return (gpu_seconds, kwh) or (0, 0) if no data ----

key_regex = re.compile(r'system\.gpu\.\d+\.powerWatts')

def process_run(run, sequential_data):
    num_gpus = get_num_gpus(run)

    power_keys_list = []
    for i, row in run.history(stream="events").iterrows():
        power_keys = {}
        for key in row.keys():
            if key_regex.search(key) and not math.isnan(row[key]):
                power_keys[key] = row[key]
                power_keys['_timestamp'] = row['_timestamp']
        if power_keys:
            power_keys_list.append(power_keys)
            try:
                sequential_data.append({
                    'run': run.name,
                    'group': run.group,
                    'timestamp': power_keys['_timestamp'],
                    **{f'GPU {g}': power_keys[f'system.gpu.{g}.powerWatts'] for g in range(8)},
                })
            except KeyError:
                pass

    if not power_keys_list:
        return 0.0, 0.0, num_gpus

    run_duration = abs(power_keys_list[-1]['_timestamp'] - power_keys_list[0]['_timestamp'])
    gpu_seconds = run_duration * num_gpus

    wattage = {}
    for i in range(1, len(power_keys_list)):
        dt = abs(power_keys_list[i]['_timestamp'] - power_keys_list[i - 1]['_timestamp'])
        for key in power_keys_list[i]:
            if key_regex.search(key):
                wattage[key] = wattage.get(key, 0.0) + power_keys_list[i][key] * dt

    node_kwh = sum(wattage.values()) / 1000 / 3600
    run_kwh = node_kwh * (num_gpus / 8)

    return gpu_seconds, run_kwh, num_gpus


# ---- Process individual runs ----

total_kwh = 0.0
total_gpu_hours = 0.0
sequential_data = []
group_summaries = []  # for final ranking

run_paths = [parse_run_url(u) for u in individual_run_urls if parse_run_url(u)]
print(f"Processing {len(run_paths)} individual runs...\n")

for rp in run_paths:
    run = api.run(rp)
    print(f"  {run.name}...", end=" ", flush=True)
    gpu_sec, kwh, num_gpus = process_run(run, sequential_data)
    total_gpu_hours += gpu_sec
    total_kwh += kwh
    gpu_hrs = gpu_sec / 3600
    print(f"{gpu_hrs:.1f} GPU-hrs, {kwh:.2f} kWh ({num_gpus} GPUs)")

ind_gpu_hrs = total_gpu_hours / 3600
ind_kwh = total_kwh
print(f"\n  Individual runs total: {ind_gpu_hrs:.1f} GPU-hrs, {ind_kwh:.2f} kWh\n")
group_summaries.append(("individual-runs", ind_gpu_hrs, ind_kwh))

# ---- Process group runs ----

print(f"Processing {len(group_urls)} group entries...\n")

for url in group_urls:
    project, group_name = parse_group_url(url)
    if not project or not group_name:
        print(f"  WARNING: Could not parse group URL: {url}")
        continue

    print(f"  Group: {group_name}")
    group_runs = list(api.runs(project, filters={"group": group_name}))
    print(f"    {len(group_runs)} runs found")

    group_gpu_sec = 0.0
    group_kwh = 0.0

    for run in group_runs:
        gpu_sec, kwh, num_gpus = process_run(run, sequential_data)
        group_gpu_sec += gpu_sec
        group_kwh += kwh

    group_gpu_hrs = group_gpu_sec / 3600
    total_gpu_hours += group_gpu_sec
    total_kwh += group_kwh

    print(f"    Total: {group_gpu_hrs:.1f} GPU-hrs, {group_kwh:.2f} kWh\n")
    group_summaries.append((group_name, group_gpu_hrs, group_kwh))

# ---- Summary ----
gpu_hrs = total_gpu_hours / 3600

print("=" * 70)
print(f"Total GPU hours: {gpu_hrs:.2f}")
print(f"Total kWh: {total_kwh:.2f}")

# Environmental impact (Jupiter cluster)
pue = 1.2
ci = 0.332   # kg CO2 / kWh
wue = 1.29   # L / kWh
embodied_co2_per_gpu_hr = 0.013  # kg
embodied_water_per_gpu_hr = 0.003  # L

energy_with_pue = total_kwh * pue
op_co2 = energy_with_pue * ci / 1000
op_water = energy_with_pue * wue / 1000
emb_co2 = embodied_co2_per_gpu_hr * gpu_hrs / 1000
emb_water = embodied_water_per_gpu_hr * gpu_hrs / 1000

print(f"\n--- Environmental Impact (Jupiter cluster) ---")
print(f"Operational CO2:  {op_co2:.2f} tCO2eq")
print(f"Operational Water: {op_water:.2f} kL")
print(f"Embodied CO2:     {emb_co2:.2f} tCO2eq")
print(f"Embodied Water:   {emb_water:.2f} kL")
print(f"TOTAL CO2:        {op_co2 + emb_co2:.2f} tCO2eq")
print(f"TOTAL Water:      {op_water + emb_water:.2f} kL")

# ---- Ranked group summary ----
print(f"\n--- Groups ranked by GPU-hours ---")
group_summaries.sort(key=lambda x: x[1], reverse=True)
for name, ghrs, kwh in group_summaries:
    print(f"  {ghrs:>10.1f} GPU-hrs  {kwh:>10.2f} kWh  {name}")

# ---- Save CSV ----
df = pd.DataFrame.from_dict(sequential_data)
print(f"\nDataFrame shape: {df.shape}")
df.to_csv("dataframes/all-runs-power.csv", index=False)
print("Saved to dataframes/all-runs-power.csv")