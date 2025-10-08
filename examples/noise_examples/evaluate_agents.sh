#!/bin/bash

# --- Config ---
SEED=69
SIM_TIME=2000 # Simulation time in seconds
NUM_EPISODES=2 # Number of episodes to average over for each matrix cell
OUTPUT_DIR="full_timeseries_results"
MATRIX_DIR="evaluation_matrix"
CONFIG_PATH="env_config/two_turbine_yaw.yaml"

# --- Agent Paths ---
PROT_CLEAN_PATH=$(ls models/protagonist_training/ts5a4m49/final_model.zip)
PROT_PROCEDURAL_PATH="models/protagonist_training/ewhkgxaj/final_model.zip"
ANTAGONIST_PATH=$(ls models/adversary_vs_pywake/*/checkpoints/model.zip)

# --- Housekeeping ---
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/*.csv # Clean old results
echo "Starting Evaluation Gauntlet with new configuration..."

# --- Run Matrix Evaluations (Agent vs. Scenario) ---
for i in $(seq 0 $(($NUM_EPISODES - 1))); do
    CURRENT_SEED=$(($SEED + $i))
    echo "--- Running evaluations for Episode $(($i + 1))/$NUM_EPISODES (Seed: $CURRENT_SEED) ---"

    # 1. PPO (Clean) Agent
    # Add --config-path to each call and use the new .pt file
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_CLEAN_PATH --scenario clean --output-path "$OUTPUT_DIR/PPO_(Clean)_in_clean_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_CLEAN_PATH --scenario procedural --output-path "$OUTPUT_DIR/PPO_(Clean)_in_procedural_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_CLEAN_PATH --scenario adversarial --antagonist-path $ANTAGONIST_PATH --output-path "$OUTPUT_DIR/PPO_(Clean)_in_adversarial_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH

    # 2. PPO (Procedural) Agent
    # Add --config-path to each call and use the new .pt file
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_PROCEDURAL_PATH --scenario clean --output-path "$OUTPUT_DIR/PPO_(Procedural)_in_clean_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_PROCEDURAL_PATH --scenario procedural --output-path "$OUTPUT_DIR/PPO_(Procedural)_in_procedural_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_PROCEDURAL_PATH --scenario adversarial --antagonist-path $ANTAGONIST_PATH --output-path "$OUTPUT_DIR/PPO_(Procedural)_in_adversarial_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH

    # 3. PyWake Agent
    # The PyWake agent doesn't need a protagonist-path, but still needs the config path
    python evaluate_agents.py --agent-type pywake --scenario clean --output-path "$OUTPUT_DIR/PyWakeAgent_in_clean_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type pywake --scenario procedural --output-path "$OUTPUT_DIR/PyWakeAgent_in_procedural_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type pywake --scenario adversarial --antagonist-path $ANTAGONIST_PATH --output-path "$OUTPUT_DIR/PyWakeAgent_in_adversarial_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
done

# --- Post-Process the Results ---
echo "--- All simulations complete. Post-processing results... ---"
python create_matrix_from_csvs.py --input-dir $OUTPUT_DIR --output-dir $MATRIX_DIR

