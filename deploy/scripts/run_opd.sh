#!/bin/bash
# run_opd.sh - Production OPD Training Wrapper with HuggingFace Upload
# Based on NeMo-RL v0.4.0 tested configurations
# 
# Usage: This script is called from the systemd service with environment variables set
#
# Required Environment Variables:
#   JOB_NAME          - Unique job identifier (e.g., opd-h200-1-32b-to-4b)
#   CONFIG_FILE       - Path to NeMo-RL config (relative to /opt/nemo-rl)
#   HF_REPO           - HuggingFace repo for upload (e.g., arunkumarvr/deepbrainz-r1-4b-math)
#   MODEL_NAME        - Model name for README (e.g., deepbrainz-r1-4b-math)
#   WANDB_API_KEY     - Weights & Biases API key
#   HF_TOKEN          - HuggingFace token
#
# Optional Environment Variables:
#   EXTRA_OVERRIDES   - Additional config overrides
#   MAX_STEPS         - Override max_num_steps (default: 2000)
#   VAL_PERIOD        - Override val_period (default: 25)
#   SAVE_PERIOD       - Override checkpointing.save_period (default: 25)

set -euo pipefail

# Configuration with defaults
JOB_NAME="${JOB_NAME:-opd-default}"
CONFIG_FILE="${CONFIG_FILE:-examples/configs/distillation_math.yaml}"
CHECKPOINT_DIR="/mnt/ckpt/${JOB_NAME}"
HF_REPO="${HF_REPO:-}"
MODEL_NAME="${MODEL_NAME:-${JOB_NAME}}"
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_PERIOD="${VAL_PERIOD:-25}"
SAVE_PERIOD="${SAVE_PERIOD:-25}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# Export for child processes
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "============================================"
log "NeMo-RL On-Policy Distillation Training"
log "============================================"
log "Job Name:        ${JOB_NAME}"
log "Config File:     ${CONFIG_FILE}"
log "Checkpoint Dir:  ${CHECKPOINT_DIR}"
log "Max Steps:       ${MAX_STEPS}"
log "Val Period:      ${VAL_PERIOD}"
log "Save Period:     ${SAVE_PERIOD}"
log "HF Repo:         ${HF_REPO:-<not set>}"
log "============================================"

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "/mnt/ckpt/logs/${JOB_NAME}"

# Change to NeMo-RL directory
cd /opt/nemo-rl

# Build command with production overrides
CMD="python examples/run_distillation_math.py"
CMD+=" --config ${CONFIG_FILE}"
CMD+=" distillation.max_num_steps=${MAX_STEPS}"
CMD+=" distillation.val_period=${VAL_PERIOD}"
CMD+=" distillation.max_val_samples=512"
CMD+=" distillation.val_at_start=true"
CMD+=" checkpointing.checkpoint_dir=${CHECKPOINT_DIR}"
CMD+=" checkpointing.save_period=${SAVE_PERIOD}"
CMD+=" checkpointing.keep_top_k=10"
CMD+=" checkpointing.metric_name=val:accuracy"
CMD+=" checkpointing.higher_is_better=true"
CMD+=" logger.wandb_enabled=true"
CMD+=" logger.wandb.project=deepbrainz-opd"
CMD+=" logger.wandb.name=${JOB_NAME}"
CMD+=" logger.log_dir=/mnt/ckpt/logs/${JOB_NAME}"
CMD+=" logger.tensorboard_enabled=true"

# Add extra overrides if provided
if [ -n "${EXTRA_OVERRIDES}" ]; then
    CMD+=" ${EXTRA_OVERRIDES}"
fi

log "Executing: ${CMD}"
log "============================================"

# Run training
TRAIN_EXIT=0
eval "${CMD}" || TRAIN_EXIT=$?

log "============================================"
log "Training completed with exit code: ${TRAIN_EXIT}"
log "============================================"

# Upload to HuggingFace if training succeeded and HF_REPO is set
if [ ${TRAIN_EXIT} -eq 0 ] && [ -n "${HF_REPO}" ]; then
    log "Training successful! Starting HuggingFace upload..."
    
    if [ -f "/home/ubuntu/upload_to_hf.sh" ]; then
        bash /home/ubuntu/upload_to_hf.sh "${CHECKPOINT_DIR}" "${HF_REPO}" "${MODEL_NAME}" || {
            log "WARNING: HuggingFace upload failed, but training succeeded"
        }
    else
        log "WARNING: upload_to_hf.sh not found, skipping upload"
    fi
elif [ ${TRAIN_EXIT} -ne 0 ]; then
    log "Training failed with exit code ${TRAIN_EXIT}. Skipping HuggingFace upload."
else
    log "HF_REPO not set, skipping HuggingFace upload."
fi

exit ${TRAIN_EXIT}
