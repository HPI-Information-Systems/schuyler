## Schuyler: Database Table Clustering Experiments

Schuyler is an experimental framework for clustering relational database tables into semantically coherent groups.
It includes multiple systems (LLM-based Schuyler, Node2Vec baseline, GPT baseline, and clustering/comdet variants), a shared experiment runner, PostgreSQL-backed schema loading, and evaluation against YAML ground truth.

This repository is structured to run reproducible scenario-based experiments across datasets such as `tpc_e`, `stack_exchange`, `adventure_works`, `magento`, and `musicbrainz`.

## What This Project Does

- Loads SQL schemas (and optionally full SQL scripts) into PostgreSQL.
- Builds table-level representations for each database scenario.
- Runs one or more clustering systems from a common experiment pipeline.
- Evaluates outputs with clustering metrics (mutual information + rand-family scores).
- Optionally logs runtime/config/metrics to Weights & Biases.

## Repository Layout

```text
schuyler/
	docker-compose.yaml            # Main Docker workflow (app + postgres)
	dockerfile                     # CUDA-based runtime image
	requirements.txt               # Python dependencies
	setup.py                       # Editable install metadata
	.env                           # API tokens / runtime secrets (local only)
	schuyler/
		database/                    # PostgreSQL abstraction and table metadata
		experimenter/                # Experiment runner + scenario/system config
			config_template.py         # Main scenarios and systems configuration
			experiment_script.py       # CLI entrypoint
			ExperimentManager.py       # Orchestrates experiments
		metrics/                     # Metric computation
		solutions/                   # Schuyler, GPT, Node2Vec, ComDet variants
```

## Requirements

- Python 3.9+ recommended (package metadata allows 3.7-3.13).
- Docker + Docker Compose for containerized runs.
- NVIDIA GPU + NVIDIA Container Toolkit for vLLM/GPU workflows.
- PostgreSQL client (`psql`) if running local DB rewrite operations outside container.

## Quick Start (Docker, Recommended)

1. Create/update `.env` in project root with your tokens:

```env
HF_TOKEN=...
WANDB_API_KEY=...
OPENAI_API_KEY=...

# Optional wandb run configuration
WANDB_PROJECT=schuyler
WANDB_ENTITY=your-team-or-user
WANDB_MODE=online
WANDB_DIR=/tmp/models
```

2. Ensure data files referenced in `schuyler/experimenter/config_template.py` are available under `/data` mount in Compose.

3. Build and run:

```bash
docker compose up --build
```

The default Compose command runs:

```bash
python3 /experiment/schuyler/experimenter/experiment_script.py --scenario base_experiment --tag runtime --wandb
```

## Configuration Guide

Primary configuration lives in `schuyler/experimenter/config_template.py`:

- `systems`: available solution families and their `train`/`test` params.
- `scenarios`: dataset-specific SQL path, schema path, and ground truth YAML path.
- `experiment_configs`: named experiment bundles consumed by CLI.

To define a new experiment:

1. Add or edit a scenario entry in `scenarios`.
2. Add or edit a system config in `systems`.
3. Register a named config under `experiment_configs`.
4. Run it via `--scenario <new_config_name>`.

## Outputs and Logging

- Metrics are computed in `schuyler/metrics/calculate_metrics.py`.
- Ground truth parsing supports hierarchical YAML clusters via `Result`.
- wandb logs include:
	- run config (system + scenario)
	- timing (`description_time`, `graph_construction_time`, etc.)
	- clustering metrics
	- produced cluster assignments

wandb configuration is environment-driven and can be changed without code edits:

- `WANDB_PROJECT`: project name (default: `schuyler`)
- `WANDB_ENTITY`: wandb entity/team (default: `Lasklu`)
- `WANDB_MODE`: wandb mode override (default: `online` with `--wandb`, otherwise `disabled`)
- `WANDB_DIR`: local wandb directory (default: `/tmp/models`)

Intermediate cache/results are written under mounted data paths (for example `/data/<database>/results/...`).

## Notes on Systems

- `SchuylerSolution`: graph + table description pipeline, optional triplet-based fine-tuning, then clustering.
- `Node2VecSolution`: graph embedding baseline with Node2Vec + AffinityPropagation.
- `GPTSolution`: schema-prompt baseline using OpenAI chat completion.
- Additional variants (`comdet`, `comdet_clustering`, `clustering`) are available through the same experiment manager.


## Citation

If you use this repository, please cite:

```bibtex
@article{10.14778/3785297.3785307,
	author = {Laskowski, Lukas and Panse, Fabian and Hladik, Michael and Portisch, Jan and Naumann, Felix},
	title = {Schuyler: Self-Supervised Clustering of Tables in Relational Databases},
	year = {2025},
	issue_date = {December 2025},
	publisher = {VLDB Endowment},
	volume = {19},
	number = {4},
	issn = {2150-8097},
	url = {https://doi.org/10.14778/3785297.3785307},
	doi = {10.14778/3785297.3785307},
	journal = {Proc. VLDB Endow.},
	month = dec,
	pages = {657--669},
	numpages = {13}
}
```

The same entry is available in `CITATION.bib`.
