import wandb

# --- Input: W&B Run URL and the summary key you want to extract ---
wandb_url = "https://wandb.ai/lasklu/schuyler/runs/wetrzow9"
summary_key = "cluster_result"

# --- Parse components from URL ---
parts = wandb_url.strip("/").split("/")
entity, project, run_id = parts[-4], parts[-3], parts[-1]

# --- Use the API to access the run ---
api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")

# --- Access the summary value ---
if summary_key in run.summary:
    print(f"{summary_key}: {run.summary[summary_key]}")
else:
    print(f"Key '{summary_key}' not found in run summary.")
