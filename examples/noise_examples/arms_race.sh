#!/bin/bash

# =============================================================================
# WindGym Training Script: Arms Race & Synthetic Self-Play (SSP)
# =============================================================================

# --- Default Configuration (can be overridden by command-line flags) ---
SSP_MODE=false
CONFIG_PATH="env_config/two_turbine_yaw.yaml"
SEED=42
PROT_DIR="models/protagonist_training"
ADV_DIR="models/adversaries_stateful"
N_ENVS=2

# Timesteps for Arms Race
AR_PROTAGONIST_TIMESTEPS=2000
AR_ADVERSARY_TIMESTEPS=2000
MAX_ITERATIONS=4

# Timesteps for Synthetic Self-Play
SSP_PROTAGONIST_TIMESTEPS=2000

# =============================================================================
# Help and Usage Function
# =============================================================================
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --ssp                     Enable Synthetic Self-Play (SSP) mode. Default: false (Arms Race mode)."
    echo "  --n-envs <num>            Number of parallel environments. Default: $N_ENVS."
    echo "  --seed <num>              Master random seed. Default: $SEED."
    echo "  --config-path <path>      Path to the environment YAML config. Default: $CONFIG_PATH."
    echo ""
    echo "Arms Race Mode Options:"
    echo "  --max-iterations <num>    Number of iterations for the arms race. Default: $MAX_ITERATIONS."
    echo "  --prot-steps <num>        Timesteps for each protagonist training cycle. Default: $AR_PROTAGONIST_TIMESTEPS."
    echo "  --adv-steps <num>         Timesteps for each adversary training cycle. Default: $AR_ADVERSARY_TIMESTEPS."
    echo ""
    echo "SSP Mode Options:"
    echo "  --ssp-steps <num>         Total timesteps for the SSP protagonist training. Default: $SSP_PROTAGONIST_TIMESTEPS."
    echo ""
    echo "  -h, --help                Display this help message."
    exit 1
}

# =============================================================================
# Helper Functions
# =============================================================================
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

find_latest_model() {
    local dir="$1"
    if [[ "$(uname)" == "Darwin" ]]; then # macOS/BSD
        find "$dir" -type f -name "*.zip" -exec stat -f "%m %N" {} + 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
    else # Assume Linux/GNU
        find "$dir" -type f -name "*.zip" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
    fi
}

count_models() {
    local dir="$1"
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
    local iteration_label="$1"; local noise_type="$2"; local antagonist_path="$3"
    log_info "=========================================="
    log_info "Training Protagonist ($iteration_label)"
    log_info "Noise type: $noise_type"
    
    local timesteps=$([ "$SSP_MODE" = true ] && echo "$SSP_PROTAGONIST_TIMESTEPS" || echo "$AR_PROTAGONIST_TIMESTEPS")
    
    local cmd="python train_protagonist.py --project-name WindGymTraining --run-name-prefix Prot_${iteration_label} --yaml-config-path $CONFIG_PATH --noise-type $noise_type --total-timesteps $timesteps --seed $SEED --n-envs $N_ENVS"
    
    if [[ "$noise_type" == "adversarial" && -n "$antagonist_path" ]]; then
        log_info "Antagonist: $antagonist_path"
        cmd="$cmd --antagonist-path $antagonist_path"
    elif [[ "$noise_type" == "synthetic_self_play" ]]; then
        log_info "Adversary Pool: $ADV_DIR"
        cmd="$cmd --adversary-pool-path $ADV_DIR"
    fi
    
    log_info "=========================================="
    log_info "Executing: $cmd"
    if ! eval "$cmd"; then log_error "Protagonist training failed."; return 1; fi
}

train_adversary() {
    local iteration="$1"; local protagonist_path="$2"
    log_info "=========================================="
    log_info "Training Adversary (Iteration $iteration)"
    log_info "Against protagonist: $protagonist_path"
    log_info "=========================================="
    local cmd="python train_adversary.py --project-name WindGymTraining --run-name-prefix Adv_Iter${iteration} --protagonist-path $protagonist_path --yaml-config-path $CONFIG_PATH --total-timesteps $AR_ADVERSARY_TIMESTEPS --seed $SEED --n-envs $N_ENVS"
    log_info "Executing: $cmd"
    if ! eval "$cmd"; then log_error "Adversary training failed."; return 1; fi
}

# =============================================================================
# Main Logic
# =============================================================================
main() {
    # --- Argument Parsing Loop ---
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --ssp) SSP_MODE=true; shift ;;
            --n-envs) N_ENVS="$2"; shift 2 ;;
            --seed) SEED="$2"; shift 2 ;;
            --config-path) CONFIG_PATH="$2"; shift 2 ;;
            --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
            --prot-steps) AR_PROTAGONIST_TIMESTEPS="$2"; shift 2 ;;
            --adv-steps) AR_ADVERSARY_TIMESTEPS="$2"; shift 2 ;;
            --ssp-steps) SSP_PROTAGONIST_TIMESTEPS="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown parameter passed: $1"; usage ;;
        esac
    done

    # --- Housekeeping ---
    mkdir -p "$PROT_DIR"
    mkdir -p "$ADV_DIR"

    if [ "$SSP_MODE" = true ]; then
        # --- Run in Synthetic Self-Play Mode ---
        log_info "Starting Synthetic Self-Play (SSP) Training"
        local adv_count=$(count_models "$ADV_DIR")
        if [[ $adv_count -eq 0 ]]; then
            log_info "No adversaries found. Training protagonist against procedural noise only."
        fi
        
        train_protagonist "SSP" "synthetic_self_play"
        if [[ $? -ne 0 ]]; then exit 1; fi
        
        log_info "Synthetic Self-Play Training Complete!"

    else
        # --- Run in Arms Race Mode ---
        log_info "Starting Arms Race Training Pipeline"
        read current_iteration next_agent < <(detect_state)
        log_info "Current state: Iteration $current_iteration, Next to train: $next_agent"

        for ((i=current_iteration; i<MAX_ITERATIONS; i++)); do
            log_info ""
            log_info "╔════════════════════════════════════════╗"
            log_info "║   Arms Race Iteration: $((i + 1)) / $MAX_ITERATIONS"
            log_info "╚════════════════════════════════════════╝"
            log_info ""
            
            if [[ "$next_agent" == "protagonist" ]]; then
                latest_adversary=$(find_latest_model "$ADV_DIR")
                if [[ $i -eq 0 ]]; then
                    # Iteration 0: Always train against procedural noise
                    noise_setting="procedural"
                    latest_adversary="" # Ensure no antagonist path is passed
                elif [[ -z "$latest_adversary" ]]; then
                    log_error "No adversary found to train against for iteration $i!"; exit 1;
                else
                    noise_setting="adversarial"
                fi
                
                train_protagonist "$i" "$noise_setting" "$latest_adversary"
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
    fi

    echo # Newline for readability
    echo "=== Training Summary ==="
    echo "Protagonists trained: $(count_models "$PROT_DIR")"
    echo "Adversaries trained: $(count_models "$ADV_DIR")"
}

# Pass all script arguments to the main function
main "$@"
