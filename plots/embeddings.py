import os
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from urllib.parse import urlparse
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib.lines import Line2D  # Import for custom legend handles

def parse_run_url(run_url):
    """
    Extracts the entity, project, and run_id from a wandb run URL.
    Expected URL format: "https://wandb.ai/<entity>/<project>/runs/<run_id>"
    Returns a tuple: (formatted_run_path, run_id) where formatted_run_path is "entity/project/run_id".
    """
    parsed = urlparse(run_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 4 and parts[2] == "runs":
        entity = parts[0]
        project = parts[1]
        run_id = parts[3]
        return f"{entity}/{project}/{run_id}", run_id
    else:
        raise ValueError(f"Invalid wandb run URL format: {run_url}")

def get_table_as_df(run, run_id, table_name, api, cache_dir="cached_artifacts"):
    """
    Retrieves the table artifact as a DataFrame.
    Expected artifact name format:
      "run-<run_id>-<table_name>:latest"
    Full artifact identifier:
      "lasklu/schuyler/run-<run_id>-<table_name>:latest"
    If a CSV exists in cache_dir, it is loaded instead.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, f"{run_id}_{table_name}.csv")
    if os.path.exists(cache_path):
        print(f"Using cached file: {cache_path}")
        return pd.read_csv(cache_path)
    else:
        print(f"Downloading artifact for {run_id} - {table_name}")
        artifact_name = f"run-{run_id}-{table_name}:latest"
        full_artifact_name = f"lasklu/schuyler/{artifact_name}"
        artifact = run.use_artifact(api.artifact(full_artifact_name))
        table = artifact.get(table_name)
        df = pd.DataFrame(data=table.data, columns=table.columns)
        df.to_csv(cache_path, index=False)
        print(f"Saved artifact to cache: {cache_path}")
        return df

def reduce_embeddings_umap(df, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Reduces the embedding dimensions using UMAP.
    Drops the non-feature columns "table" and "label" (the latter is used only for coloring).
    """
    features = df.drop(columns=["table", "label"], errors="ignore")
    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced = reducer.fit_transform(features)
    return reduced

def plot_embeddings(ax, df, reduced, title, quality_text):
    """
    Plots the 2D embeddings on the provided axis.
    Colors the points by the "label" column.
    Adds the title (database name and tags) and quality scores to the subplot.
    The legend is omitted.
    """
    labels = df["label"]
    unique_labels = labels.unique()
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    for idx, label in enumerate(unique_labels):
        idxs = (labels == label)
        ax.scatter(reduced[idxs, 0], reduced[idxs, 1], color=cmap(idx), alpha=0.7)
    
    # Set a larger title.
    ax.set_title(title, fontsize=22)
    # Remove tick labels so that no axis labels appear.
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust quality text position based on the title.
    if title == "Database aware":
        x = 0.42
        y = 0.15
    elif title == "Neighbor":
        x = 0.42
        y = 0.95
    else:
        x = 0.95
        y = 0.95
    ax.text(x, y, quality_text,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=16,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# List of provided run URLs.
run_urls = [
    "https://wandb.ai/lasklu/schuyler/runs/q8lh0j9j",
    "https://wandb.ai/lasklu/schuyler/runs/reoeelow",
    "https://wandb.ai/lasklu/schuyler/runs/aht73kef",
    "https://wandb.ai/lasklu/schuyler/runs/issvjot1"
]

# Initialize the wandb API.
api = wandb.Api()

# List to store tuples: (database name, DataFrame, UMAP reduced data, tags string, quality scores text)
runs_data = []

for run_url in run_urls:
    print(f"\nProcessing run: {run_url}")
    run_path, run_id = parse_run_url(run_url)
    print("Run path:", run_path)
    
    # Get the run object.
    run = api.run(run_path)
    
    # Retrieve the database name from the run's config.
    db_name = run.config.get("database_name", run_id)
    
    # Retrieve the run's tags (as a list) and join them.
    tags = run.tags if hasattr(run, "tags") and run.tags else []
    tags_str = ", ".join(tags) if tags else "No tags"
    
    try:
        # Retrieve (and cache) the after_finetuning table.
        df_after = get_table_as_df(run, run_id, "embedding_table_after_finetuning", api)
    except Exception as e:
        print(f"Error loading table for run {run_id}: {e}")
        continue
    
    print("After finetuning DataFrame head:")
    print(df_after.head())
    
    # Reduce embeddings using UMAP.
    reduced = reduce_embeddings_umap(df_after, n_components=2)
    
    # Compute quality scores using the "label" column as cluster labels.
    labels = df_after["label"]
    unique_labels = labels.unique()
    if len(unique_labels) > 1:
        sil = silhouette_score(reduced, labels)
        ch = calinski_harabasz_score(reduced, labels)
        db = davies_bouldin_score(reduced, labels)
    else:
        sil = ch = db = None
    
    if sil is not None:
        quality_text = f"Silhouette: {sil:.2f}\nCH: {ch:.2f}\nDB: {db:.2f}"
        quality_text = f"Silhouette: {sil:.2f}"
    else:
        quality_text = "Silhouette: N/A\nCH: N/A\nDB: N/A"
    print(f"Quality scores for run {run_id}: {quality_text.replace(chr(10), ', ')}")
    
    runs_data.append((db_name, df_after, reduced, tags_str, quality_text))

# Create a figure with a 2x2 grid of subplots.
n_runs = len(runs_data)
if n_runs == 0:
    print("No runs processed successfully.")
else:
    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 7))
    axes = axes.flatten()
    
    for idx, (db_name, df, reduced, tags_str, quality_text) in enumerate(runs_data):
        ax = axes[idx]
        # Use custom titles based on the tags.
        if tags_str == "description":
            title = "Database aware"
        elif tags_str == "prompt_description_neighbor":
            title = "Neighbor"
        elif tags_str == "prompt_description_random":
            title = "Random"
        elif tags_str == "prompt_description_similarity":
            title = "Similarity"
        else:
            title = f"{db_name}\n{tags_str}"
        plot_embeddings(ax, df, reduced, title, quality_text)
    
    # Turn off any remaining subplots if there are fewer than 4 runs.
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    
    # Create a joint legend below all the subplots.
    # We assume that all runs have the same labels, so we use the labels from the first run.
    first_df = runs_data[0][1]
    unique_labels = sorted(first_df["label"].unique())  # Sorted list of unique labels
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=str(label))
        for i, label in enumerate(unique_labels)
    ]
    # Place the legend in a new, lower axis below the subplots.
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(unique_labels), bbox_to_anchor=(0.5, 0), fontsize=14)
    
    # Adjust layout to accommodate the joint legend.
    plt.tight_layout(rect=[-0.05, 0.05, 1, 0.95])
    # Save as a PDF file.
    output_filename = "embedding_comparison_after_finetuning_umap.pdf"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Saved combined UMAP plot to {output_filename}")
    plt.close(fig)
