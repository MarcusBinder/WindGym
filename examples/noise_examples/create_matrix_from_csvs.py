import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    input_dir: str = "full_timeseries_results"
    """Directory containing the CSV files from the evaluation runs."""
    output_dir: str = "evaluation_matrix"
    """Directory to save the final matrix CSV and heatmap plot."""


def parse_filename(filename):
    """Parses 'AGENT_in_SCENARIO_epNUM.csv' to get agent, scenario, and episode."""
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_in_")
    if len(parts) != 2:
        return None, None, None

    agent_name = parts[0].replace("_", " ")

    env_episode_part = parts[1]
    ep_parts = env_episode_part.split("_ep")

    env_name = ep_parts[0].replace("_", " ").title()
    episode_num = int(ep_parts[1]) if len(ep_parts) > 1 else 0

    return agent_name, env_name, episode_num


def calculate_power_gain(filepath):
    """Loads a CSV and computes the total power gain vs. baseline for the episode."""
    df = pd.read_csv(filepath)
    if "power_agent" not in df.columns or "power_baseline" not in df.columns:
        return np.nan

    total_agent_power = df["power_agent"].mean()
    total_baseline_power = df["power_baseline"].mean()

    if total_baseline_power == 0:
        return np.nan

    return (total_agent_power / total_baseline_power) - 1


def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    # {(agent, env): [list_of_gains]}
    results = {}

    for filepath in csv_files:
        agent, env, episode = parse_filename(filepath)
        if agent is None:
            continue

        gain = calculate_power_gain(filepath)
        key = (agent, env)
        if key not in results:
            results[key] = []
        results[key].append(gain)

    # --- Create the results matrix ---
    agents = sorted(list(set(key[0] for key in results.keys())))
    environments = sorted(list(set(key[1] for key in results.keys())))

    matrix_df = pd.DataFrame(index=agents, columns=environments, dtype=str)
    numeric_matrix = pd.DataFrame(index=agents, columns=environments, dtype=float)

    for (agent, env), gains in results.items():
        mean_gain = np.mean(gains)
        std_gain = np.std(gains)
        matrix_df.loc[agent, env] = f"{mean_gain:.3f} ± {std_gain:.3f}"
        numeric_matrix.loc[agent, env] = mean_gain

    matrix_csv_path = os.path.join(args.output_dir, "power_gain_matrix.csv")
    matrix_df.to_csv(matrix_csv_path)

    print("\n" + "=" * 25 + " GAUNTLET RESULTS (Power Gain vs Baseline) " + "=" * 25)
    print(matrix_df)
    print("=" * 82)

    # --- Create and save the heatmap ---
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        numeric_matrix,
        annot=matrix_df.values,
        fmt="s",
        cmap="viridis",
        ax=ax,
        cbar=True,
        linewidths=0.5,
        cbar_kws={"label": "Mean Episodic Power Gain vs Baseline"},
        annot_kws={"size": 12},
    )
    ax.set_title(
        "Agent Robustness Evaluation Matrix", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlabel("Evaluation Environment", fontsize=12)
    ax.set_ylabel("Protagonist Agent", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "power_gain_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Analysis complete. Matrix and heatmap saved to '{args.output_dir}'")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
