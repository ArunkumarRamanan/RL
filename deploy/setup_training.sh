#!/bin/bash
# setup_training.sh - Master deployment script for NeMo-RL Training
#
# This script sets up a complete OPD/GRPO training environment on an EC2 instance.
# Based on ACTUAL tested configs from the NeMo-RL repository.
#
# Usage:
#   ./setup_training.sh <instance_type> <wandb_api_key> <hf_token>
#
# Arguments:
#   instance_type: One of: h200-1, h200-2, a100-80, a100-40
#   wandb_api_key: Your Weights & Biases API key
#   hf_token:      Your HuggingFace token
#
# Instance Configurations (ALL UNIQUE):
#   h200-1:  OPD  - Qwen3-32B → Qwen3-4B-Base
#   h200-2:  OPD  - Qwen3-32B → Qwen3-1.7B-Base  
#   a100-80: OPD  - Qwen3-4B  → Qwen3-1.7B-Base
#   a100-40: GRPO - Qwen3-8B-Base (RL training)

set -euo pipefail

# ============================================
# Configuration
# ============================================
INSTANCE_TYPE="${1:-}"
WANDB_API_KEY="${2:-}"
HF_TOKEN="${3:-}"

# Validate arguments
if [ -z "${INSTANCE_TYPE}" ] || [ -z "${WANDB_API_KEY}" ] || [ -z "${HF_TOKEN}" ]; then
    echo "Usage: $0 <instance_type> <wandb_api_key> <hf_token>"
    echo ""
    echo "Instance types (ALL UNIQUE configurations):"
    echo "  h200-1   - OPD:  Qwen3-32B → Qwen3-4B-Base"
    echo "  h200-2   - OPD:  Qwen3-32B → Qwen3-1.7B-Base"
    echo "  a100-80  - OPD:  Qwen3-4B  → Qwen3-1.7B-Base"
    echo "  a100-40  - GRPO: Qwen3-8B-Base (RL training)"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "============================================"
log "NeMo-RL Training Setup"
log "============================================"
log "Instance Type: ${INSTANCE_TYPE}"
log "============================================"

# ============================================
# Instance-specific configuration
# Based on ACTUAL NeMo-RL tested configs
# ============================================
case "${INSTANCE_TYPE}" in
    h200-1)
        # OPD: Qwen3-32B → Qwen3-4B-Base
        # Config: distillation-qwen3-32b-to-4b-base-1n8g-fsdp2tp2-dynamicbatch.v1.yaml
        ALGORITHM="distillation"
        JOB_NAME="opd-h200-1-32b-to-4b"
        CONFIG_FILE="examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-1n8g-fsdp2tp2-dynamicbatch.v1.yaml"
        HF_REPO="arunkumarvr/deepbrainz-r1-4b-math-reasoning"
        MODEL_NAME="deepbrainz-r1-4b-math-reasoning"
        MAX_STEPS="2000"
        EXTRA_OVERRIDES=""
        ENTRY_SCRIPT="examples/run_distillation_math.py"
        ;;
    h200-2)
        # OPD: Qwen3-32B → Qwen3-1.7B-Base (DIFFERENT student from h200-1)
        # Config: distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1.yaml
        ALGORITHM="distillation"
        JOB_NAME="opd-h200-2-32b-to-1.7b"
        CONFIG_FILE="examples/configs/recipes/llm/distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1.yaml"
        HF_REPO="arunkumarvr/deepbrainz-r1-1.7b-math-reasoning"
        MODEL_NAME="deepbrainz-r1-1.7b-math-reasoning"
        MAX_STEPS="2500"
        EXTRA_OVERRIDES=""
        ENTRY_SCRIPT="examples/run_distillation_math.py"
        ;;
    a100-80)
        # OPD: Qwen3-4B → Qwen3-1.7B-Base (DIFFERENT teacher from h200 instances)
        # Config: distillation_math.yaml (default tested config)
        ALGORITHM="distillation"
        JOB_NAME="opd-a100-80-4b-to-1.7b"
        CONFIG_FILE="examples/configs/distillation_math.yaml"
        HF_REPO="arunkumarvr/deepbrainz-r1-1.7b-math-4b-teacher"
        MODEL_NAME="deepbrainz-r1-1.7b-math-4b-teacher"
        MAX_STEPS="3000"
        EXTRA_OVERRIDES=""
        ENTRY_SCRIPT="examples/run_distillation_math.py"
        ;;
    a100-40)
        # GRPO: Qwen3-8B-Base (RL training, NOT distillation)
        # Config: grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.yaml
        ALGORITHM="grpo"
        JOB_NAME="grpo-a100-40-8b-base"
        CONFIG_FILE="examples/configs/recipes/llm/grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.yaml"
        HF_REPO="arunkumarvr/deepbrainz-r1-8b-math-grpo"
        MODEL_NAME="deepbrainz-r1-8b-math-grpo"
        MAX_STEPS="1000"
        # Enable checkpointing (disabled by default in this config)
        EXTRA_OVERRIDES="checkpointing.enabled=true checkpointing.checkpoint_dir=/mnt/ckpt/${JOB_NAME} policy.generation.vllm_cfg.gpu_memory_utilization=0.4"
        ENTRY_SCRIPT="examples/run_grpo_math.py"
        ;;
    *)
        log "ERROR: Unknown instance type: ${INSTANCE_TYPE}"
        log "Valid types: h200-1, h200-2, a100-80, a100-40"
        exit 1
        ;;
esac

log "Algorithm:  ${ALGORITHM}"
log "Job Name:   ${JOB_NAME}"
log "Config:     ${CONFIG_FILE}"
log "HF Repo:    ${HF_REPO}"
log "Max Steps:  ${MAX_STEPS}"
log "Entry:      ${ENTRY_SCRIPT}"

# ============================================
# Step 1: Create directories
# ============================================
log "Creating directories..."
sudo mkdir -p /mnt/ckpt
sudo mkdir -p /mnt/ckpt/logs
sudo chown -R ubuntu:ubuntu /mnt/ckpt 2>/dev/null || true

# ============================================
# Step 2: Create training script
# ============================================
log "Creating training script..."
sudo tee /home/ubuntu/run_training.sh > /dev/null << 'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

ALGORITHM="${ALGORITHM:-distillation}"
JOB_NAME="${JOB_NAME:-training-default}"
CONFIG_FILE="${CONFIG_FILE:-examples/configs/distillation_math.yaml}"
ENTRY_SCRIPT="${ENTRY_SCRIPT:-examples/run_distillation_math.py}"
CHECKPOINT_DIR="/mnt/ckpt/${JOB_NAME}"
HF_REPO="${HF_REPO:-}"
MODEL_NAME="${MODEL_NAME:-${JOB_NAME}}"
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_PERIOD="${VAL_PERIOD:-25}"
SAVE_PERIOD="${SAVE_PERIOD:-25}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "============================================"
log "NeMo-RL ${ALGORITHM^^} Training"
log "============================================"
log "Job: ${JOB_NAME}"
log "Config: ${CONFIG_FILE}"
log "Entry: ${ENTRY_SCRIPT}"
log "Checkpoint: ${CHECKPOINT_DIR}"
log "============================================"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "/mnt/ckpt/logs/${JOB_NAME}"

cd /opt/nemo-rl

# Build command based on algorithm
CMD="python ${ENTRY_SCRIPT}"
CMD+=" --config ${CONFIG_FILE}"

if [ "${ALGORITHM}" = "distillation" ]; then
    CMD+=" distillation.max_num_steps=${MAX_STEPS}"
    CMD+=" distillation.val_period=${VAL_PERIOD}"
    CMD+=" distillation.max_val_samples=512"
    CMD+=" distillation.val_at_start=true"
elif [ "${ALGORITHM}" = "grpo" ]; then
    CMD+=" grpo.max_num_steps=${MAX_STEPS}"
    CMD+=" grpo.val_period=${VAL_PERIOD}"
fi

CMD+=" checkpointing.checkpoint_dir=${CHECKPOINT_DIR}"
CMD+=" checkpointing.save_period=${SAVE_PERIOD}"
CMD+=" checkpointing.keep_top_k=10"
CMD+=" logger.wandb_enabled=true"
CMD+=" logger.wandb.project=deepbrainz-${ALGORITHM}"
CMD+=" logger.wandb.name=${JOB_NAME}"
CMD+=" logger.log_dir=/mnt/ckpt/logs/${JOB_NAME}"
CMD+=" logger.tensorboard_enabled=true"

if [ -n "${EXTRA_OVERRIDES}" ]; then
    CMD+=" ${EXTRA_OVERRIDES}"
fi

log "Running: ${CMD}"
TRAIN_EXIT=0
eval "${CMD}" || TRAIN_EXIT=$?

if [ ${TRAIN_EXIT} -eq 0 ] && [ -n "${HF_REPO}" ]; then
    log "Training complete! Uploading to HuggingFace..."
    bash /home/ubuntu/upload_to_hf.sh "${CHECKPOINT_DIR}" "${HF_REPO}" "${MODEL_NAME}" || true
fi

exit ${TRAIN_EXIT}
SCRIPT_EOF
sudo chmod +x /home/ubuntu/run_training.sh

# ============================================
# Step 3: Create HuggingFace upload script
# ============================================
log "Creating HuggingFace upload script..."
sudo tee /home/ubuntu/upload_to_hf.sh > /dev/null << 'UPLOAD_EOF'
#!/bin/bash
set -euo pipefail

CHECKPOINT_DIR="${1:-}"
HF_REPO="${2:-}"
MODEL_NAME="${3:-}"

if [ -z "${CHECKPOINT_DIR}" ] || [ -z "${HF_REPO}" ] || [ -z "${MODEL_NAME}" ]; then
    echo "Usage: $0 <checkpoint_dir> <hf_repo> <model_name>"
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [HF-UPLOAD] $*"; }

log "Finding best checkpoint in ${CHECKPOINT_DIR}..."
BEST_CKPT=""
for ckpt in $(ls -td "${CHECKPOINT_DIR}"/step_* 2>/dev/null); do
    if [ -d "${ckpt}/policy/weights" ]; then
        BEST_CKPT="${ckpt}"
        break
    fi
done

if [ -z "${BEST_CKPT}" ]; then
    log "ERROR: No checkpoint found"
    exit 1
fi

log "Using checkpoint: ${BEST_CKPT}"

CONVERTED_DIR="/tmp/hf_upload_$(date +%s)"
mkdir -p "${CONVERTED_DIR}"

cd /opt/nemo-rl
python examples/converters/convert_dcp_to_hf.py \
    --config "${BEST_CKPT}/config.yaml" \
    --dcp-ckpt-path "${BEST_CKPT}/policy/weights" \
    --hf-ckpt-path "${CONVERTED_DIR}"

if [ -d "${BEST_CKPT}/policy/tokenizer" ]; then
    rsync -ahP "${BEST_CKPT}/policy/tokenizer/" "${CONVERTED_DIR}/"
fi

STEP_NUM=$(basename "${BEST_CKPT}" | sed 's/step_//')

cat > "${CONVERTED_DIR}/README.md" << EOF
---
license: apache-2.0
language: [en]
tags: [math, reasoning, nemo-rl, qwen3, deepbrainz]
pipeline_tag: text-generation
---
# ${MODEL_NAME}
Trained using NVIDIA NeMo-RL v0.4.0.
Training steps: ${STEP_NUM}
EOF

pip install -q huggingface_hub 2>/dev/null || true
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
python -c "from huggingface_hub import HfApi; HfApi().create_repo('${HF_REPO}', exist_ok=True)" 2>/dev/null || true
huggingface-cli upload "${HF_REPO}" "${CONVERTED_DIR}" . --commit-message "Upload ${MODEL_NAME} step ${STEP_NUM}"

log "Uploaded to: https://huggingface.co/${HF_REPO}"
rm -rf "${CONVERTED_DIR}"
UPLOAD_EOF
sudo chmod +x /home/ubuntu/upload_to_hf.sh

# ============================================
# Step 4: Create spot monitor script
# ============================================
log "Creating spot monitor script..."
sudo tee /home/ubuntu/spot_monitor.sh > /dev/null << 'SPOT_EOF'
#!/bin/bash
set -uo pipefail

CONTAINER_NAME="${1:-nemo-training}"
METADATA_URL="http://169.254.169.254/latest/meta-data/spot/termination-time"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SPOT] $*"; }
log "Started monitoring container: ${CONTAINER_NAME}"

while true; do
    RESPONSE=$(curl -s --connect-timeout 2 "${METADATA_URL}" 2>/dev/null || echo "")
    if echo "${RESPONSE}" | grep -qE "^[0-9]{4}-[0-9]{2}-[0-9]{2}T"; then
        log "TERMINATION DETECTED: ${RESPONSE}"
        docker exec "${CONTAINER_NAME}" pkill -TERM -f "python.*" 2>/dev/null || true
        sleep 90
        docker stop -t 30 "${CONTAINER_NAME}" 2>/dev/null || true
        exit 0
    fi
    sleep 5
done
SPOT_EOF
sudo chmod +x /home/ubuntu/spot_monitor.sh

# ============================================
# Step 5: Create systemd services
# ============================================
log "Creating systemd services..."

# Determine container name based on algorithm
CONTAINER_NAME="nemo-${ALGORITHM}"

# Main training service
sudo tee /etc/systemd/system/nemo-training.service > /dev/null << SVCEOF
[Unit]
Description=NeMo-RL ${ALGORITHM^^} Training - ${JOB_NAME}
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=root
Restart=on-failure
RestartSec=60
TimeoutStartSec=900
TimeoutStopSec=180

Environment="ALGORITHM=${ALGORITHM}"
Environment="JOB_NAME=${JOB_NAME}"
Environment="CONFIG_FILE=${CONFIG_FILE}"
Environment="ENTRY_SCRIPT=${ENTRY_SCRIPT}"
Environment="HF_REPO=${HF_REPO}"
Environment="MODEL_NAME=${MODEL_NAME}"
Environment="MAX_STEPS=${MAX_STEPS}"
Environment="VAL_PERIOD=25"
Environment="SAVE_PERIOD=25"
Environment="EXTRA_OVERRIDES=${EXTRA_OVERRIDES}"
Environment="WANDB_API_KEY=${WANDB_API_KEY}"
Environment="HF_TOKEN=${HF_TOKEN}"
Environment="CONTAINER_NAME=${CONTAINER_NAME}"
Environment="NRL_IMAGE=nvcr.io/nvidia/nemo-rl:v0.4.0"

ExecStartPre=-/usr/bin/docker rm -f ${CONTAINER_NAME}
ExecStart=/usr/bin/docker run \\
    --name ${CONTAINER_NAME} \\
    --gpus all \\
    --ipc=host \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --network=host \\
    -e NVIDIA_VISIBLE_DEVICES=all \\
    -e ALGORITHM \\
    -e WANDB_API_KEY \\
    -e HF_TOKEN \\
    -e JOB_NAME \\
    -e CONFIG_FILE \\
    -e ENTRY_SCRIPT \\
    -e HF_REPO \\
    -e MODEL_NAME \\
    -e MAX_STEPS \\
    -e VAL_PERIOD \\
    -e SAVE_PERIOD \\
    -e EXTRA_OVERRIDES \\
    -v /mnt/ckpt:/mnt/ckpt \\
    -v /home/ubuntu/run_training.sh:/home/ubuntu/run_training.sh:ro \\
    -v /home/ubuntu/upload_to_hf.sh:/home/ubuntu/upload_to_hf.sh:ro \\
    -w /opt/nemo-rl \\
    nvcr.io/nvidia/nemo-rl:v0.4.0 \\
    bash /home/ubuntu/run_training.sh

ExecStop=/usr/bin/docker stop -t 120 ${CONTAINER_NAME}

[Install]
WantedBy=multi-user.target
SVCEOF

# Spot monitor service
sudo tee /etc/systemd/system/spot-monitor.service > /dev/null << SPOTEOF
[Unit]
Description=EC2 Spot Termination Monitor
After=nemo-training.service

[Service]
Type=simple
User=root
Restart=always
RestartSec=10
ExecStart=/home/ubuntu/spot_monitor.sh ${CONTAINER_NAME}

[Install]
WantedBy=multi-user.target
SPOTEOF

# ============================================
# Step 6: Enable and start services
# ============================================
log "Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable nemo-training.service
sudo systemctl enable spot-monitor.service

# ============================================
# Step 7: Pull container image
# ============================================
log "Pulling NeMo-RL container image..."
sudo docker pull nvcr.io/nvidia/nemo-rl:v0.4.0 || {
    log "WARNING: Could not pull container image. Will pull on first start."
}

# ============================================
# Summary
# ============================================
log "============================================"
log "Setup Complete!"
log "============================================"
log ""
log "Configuration:"
log "  Algorithm:  ${ALGORITHM}"
log "  Job Name:   ${JOB_NAME}"
log "  Config:     ${CONFIG_FILE}"
log "  HF Repo:    ${HF_REPO}"
log "  Max Steps:  ${MAX_STEPS}"
log ""
log "Commands:"
log "  Start training:    sudo systemctl start nemo-training"
log "  Stop training:     sudo systemctl stop nemo-training"
log "  View logs:         sudo journalctl -u nemo-training -f"
log "  Check status:      sudo systemctl status nemo-training"
log ""
log "Checkpoints: /mnt/ckpt/${JOB_NAME}"
log "WandB:       deepbrainz-${ALGORITHM}"
log "============================================"

# Ask if user wants to start now
read -p "Start training now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Starting training..."
    sudo systemctl start spot-monitor.service
    sudo systemctl start nemo-training.service
    log "Training started! Use 'sudo journalctl -u nemo-training -f' to view logs."
else
    log "Training not started. Run 'sudo systemctl start nemo-training' when ready."
fi
