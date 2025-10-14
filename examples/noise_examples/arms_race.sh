#!/bin/bash

# =============================================================================
# Arms Race Training Script (Platform-Aware Final Version)
# =============================================================================

# --- Configuration ---
CONFIG_PATH="env_config/two_turbine_yaw.yaml"
MAX_ITERATIONS=4
SEED=42
PROTAGONIST_TIMESTEPS=900
ADVERSARY_TIMESTEPS=900
PROT_DIR="models/protagonist_training"
ADV_DIR="models/adversaries_stateful"

mkdir -p "$PROT_DIR"
mkdir -p "$ADV_DIR"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    # Redirect echo to standard error (>&2) to separate logs from function output.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

find_latest_model() {
    local dir="$1"
    
    # --- PLATFORM-AWARE FIX ---
    # This block detects the OS and uses the correct command.
    # macOS/BSD `find` does not support `-printf`. We use `stat` instead.
    # Linux `find` uses the efficient `-printf`.
    if [[ "$(uname)" == "Darwin" ]]; then # Darwin is the kernel name for macOS
        find "$dir" -type f -name "*.zip" -exec stat -f "%m %N" {} + 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
    else # Assume Linux/GNU
        find "$dir" -type f -name "*.zip" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
    fi
}

count_models() {
    local dir="$1"
    # This simple find command is compatible with all systems.
    find "$dir" -type f -name "*.zip" 2>/dev/null | wc -l | tr -d ' '
}

detect_state() {
    local prot_count=$(count_models "$PROT_DIR")
    local adv_count=$(count_models "$ADV_DIR")
    log_info "Detected $prot_count protagonist(s) and $adv_count adversary model(s)"
    local iteration=$((prot_count + adv_count))
    local next_agent
    if [[ $prot_count -gt $adv_count ]]; then
        next_agent="adversary"
    else
        next_agent="protagonist"
    fi
    echo "$iteration $next_agent"
}

# =============================================================================
# Training Functions
# =============================================================================

train_protagonist() {
    local iteration="$1"; local noise_type="$2"; local antagonist_path="$3"
    log_info "=========================================="
    log_info "Training Protagonist (Iteration $iteration)"
    log_info "Noise type: $noise_type"
    local cmd="python train_protagonist.py --project-name ArmsRace --run-name-prefix Prot_Iter${iteration} --yaml-config-path $CONFIG_PATH --noise-type $noise_type --total-timesteps $PROTAGONIST_TIMESTEPS --seed $SEED"
    if [[ "$noise_type" == "adversarial" && -n "$antagonist_path" ]]; then
        log_info "Antagonist: $antagonist_path"
        cmd="$cmd --antagonist-path $antagonist_path"
    fi
    log_info "=========================================="
    log_info "Executing: $cmd"
    if ! eval "$cmd"; then
        log_error "Protagonist training failed."
        return 1
    fi
}

train_adversary() {
    local iteration="$1"; local protagonist_path="$2"
    log_info "=========================================="
    log_info "Training Adversary (Iteration $iteration)"
    log_info "Against protagonist: $protagonist_path"
    log_info "=========================================="
    local cmd="python train_adversary.py --project-name ArmsRace --run-name-prefix Adv_Iter${iteration} --protagonist-path $protagonist_path --yaml-config-path $CONFIG_PATH --total-timesteps $ADVERSARY_TIMESTEPS --seed $SEED"
    log_info "Executing: $cmd"
    if ! eval "$cmd"; then
        log_error "Adversary training failed."
        return 1
    fi
}

# =============================================================================
# Main Training Loop
# =============================================================================
main() {
    log_info "Starting Arms Race Training Pipeline"
    read current_iteration next_agent < <(detect_state)
    log_info "Current state: Iteration $current_iteration, Next: $next_agent"

    for ((i=current_iteration; i<MAX_ITERATIONS; i++)); do
        log_info ""
        log_info "╔════════════════════════════════════════╗"
        log_info "║   Arms Race Iteration: $((i + 1)) / $MAX_ITERATIONS"
        log_info "╚════════════════════════════════════════╝"
        log_info ""
        if [[ "$next_agent" == "protagonist" ]]; then
            latest_adversary=$(find_latest_model "$ADV_DIR")
            if [[ $i -ne 0 && -z "$latest_adversary" ]]; then
                log_error "No adversary found to train against!"; exit 1;
            fi
            train_protagonist "$i" "${latest_adversary:+adversarial}" "$latest_adversary"
            if [[ $? -ne 0 ]]; then exit 1; fi
            next_agent="adversary"
        else
            latest_protagonist=$(find_latest_model "$PROT_DIR")
            if [[ -z "$latest_protagonist" ]]; then
                log_error "No protagonist found to train against!"; exit 1;
            fi
            train_adversary "$i" "$latest_protagonist"
            if [[ $? -ne 0 ]]; then exit 1; fi
            next_agent="protagonist"
        fi
        sleep 2
    done

    log_info "Arms Race Training Complete!"
    echo # Newline for readability
    echo "=== Training Summary ==="
    echo "Protagonists trained: $(count_models "$PROT_DIR")"
    echo "Adversaries trained: $(count_models "$ADV_DIR")"
}

# Run main, exit if it fails
main
