#!/bin/bash
# upload_to_hf.sh - Upload trained OPD model to HuggingFace
# Based on NeMo-RL checkpoint converter: examples/converters/convert_dcp_to_hf.py
#
# Usage: ./upload_to_hf.sh <checkpoint_dir> <hf_repo> <model_name>
#
# This script:
# 1. Finds the best checkpoint in the directory
# 2. Converts DCP format to HuggingFace format
# 3. Generates a model card (README.md)
# 4. Uploads to HuggingFace Hub

set -euo pipefail

# Arguments
CHECKPOINT_DIR="${1:-}"
HF_REPO="${2:-}"
MODEL_NAME="${3:-}"

# Validate arguments
if [ -z "${CHECKPOINT_DIR}" ] || [ -z "${HF_REPO}" ] || [ -z "${MODEL_NAME}" ]; then
    echo "Usage: $0 <checkpoint_dir> <hf_repo> <model_name>"
    echo "Example: $0 /mnt/ckpt/opd-h200-1 arunkumarvr/deepbrainz-r1-4b-math deepbrainz-r1-4b-math"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "============================================"
log "HuggingFace Model Upload"
log "============================================"
log "Checkpoint Dir: ${CHECKPOINT_DIR}"
log "HF Repo:        ${HF_REPO}"
log "Model Name:     ${MODEL_NAME}"
log "============================================"

# Find best checkpoint (latest step_* directory with weights)
BEST_CKPT=""
for ckpt in $(ls -td "${CHECKPOINT_DIR}"/step_* 2>/dev/null); do
    if [ -d "${ckpt}/policy/weights" ]; then
        BEST_CKPT="${ckpt}"
        break
    fi
done

if [ -z "${BEST_CKPT}" ]; then
    log "ERROR: No valid checkpoint found in ${CHECKPOINT_DIR}"
    log "Looking for: ${CHECKPOINT_DIR}/step_*/policy/weights"
    ls -la "${CHECKPOINT_DIR}" 2>/dev/null || true
    exit 1
fi

log "Found best checkpoint: ${BEST_CKPT}"

# Check for config.yaml
if [ ! -f "${BEST_CKPT}/config.yaml" ]; then
    log "ERROR: config.yaml not found in ${BEST_CKPT}"
    exit 1
fi

# Create temporary directory for converted model
CONVERTED_DIR="/tmp/hf_upload_${MODEL_NAME}_$(date +%s)"
rm -rf "${CONVERTED_DIR}"
mkdir -p "${CONVERTED_DIR}"

log "Converting checkpoint to HuggingFace format..."

# Run NeMo-RL DCP to HF converter
cd /opt/nemo-rl
python examples/converters/convert_dcp_to_hf.py \
    --config "${BEST_CKPT}/config.yaml" \
    --dcp-ckpt-path "${BEST_CKPT}/policy/weights" \
    --hf-ckpt-path "${CONVERTED_DIR}"

# Copy tokenizer if it exists
if [ -d "${BEST_CKPT}/policy/tokenizer" ]; then
    log "Copying tokenizer..."
    rsync -ahP "${BEST_CKPT}/policy/tokenizer/" "${CONVERTED_DIR}/"
fi

# Extract metadata from config for README
STUDENT_MODEL=$(grep -oP 'model_name:\s*\K[^\s]+' "${BEST_CKPT}/config.yaml" | head -1 || echo "Unknown")
TEACHER_MODEL=$(grep -A5 '^teacher:' "${BEST_CKPT}/config.yaml" | grep -oP 'model_name:\s*\K[^\s]+' | head -1 || echo "Unknown")
SEQ_LENGTH=$(grep -oP 'max_total_sequence_length:\s*\K[0-9]+' "${BEST_CKPT}/config.yaml" | head -1 || echo "8192")

# Get step number from checkpoint path
STEP_NUM=$(basename "${BEST_CKPT}" | sed 's/step_//')

log "Generating model card..."

# Generate README.md with model card
cat > "${CONVERTED_DIR}/README.md" << EOF
---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- math
- reasoning
- distillation
- on-policy-distillation
- nemo-rl
- qwen3
- deepbrainz
base_model: ${STUDENT_MODEL}
datasets:
- nvidia/OpenMathInstruct-2
- agentica-org/DeepScaleR-1.5B-Preview
pipeline_tag: text-generation
model-index:
- name: ${MODEL_NAME}
  results:
  - task:
      type: text-generation
      name: Mathematical Reasoning
    dataset:
      type: custom
      name: AIME2024
    metrics:
    - type: accuracy
      name: Validation Accuracy
      value: TBD
---

# ${MODEL_NAME}

A small language model optimized for **mathematical reasoning**, trained using **On-Policy Distillation (OPD)** with the NVIDIA NeMo-RL framework.

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | \`${STUDENT_MODEL}\` |
| **Teacher Model** | \`${TEACHER_MODEL}\` |
| **Training Method** | On-Policy Distillation |
| **Framework** | [NVIDIA NeMo-RL v0.4.0](https://github.com/NVIDIA-NeMo/RL) |
| **Training Dataset** | DeepScaler / OpenMathInstruct-2 |
| **Max Sequence Length** | ${SEQ_LENGTH} tokens |
| **Training Steps** | ${STEP_NUM} |

## Training Configuration

### On-Policy Distillation

On-Policy Distillation generates responses from the student model, then aligns the student's output distribution to match the teacher's via KL divergence. This approach:

- Trains the student on its **own distribution** (on-policy)
- Uses **reverse KL** for mode-seeking behavior
- Enables knowledge transfer without requiring teacher-generated data

### Key Parameters

\`\`\`yaml
# Distillation Config
distillation:
  num_prompts_per_step: 64
  topk_logits_k: 64
  
loss_fn:
  kl_type: reverse  # Reverse KL for mode-seeking behavior

# Models
policy:
  model_name: ${STUDENT_MODEL}
  max_total_sequence_length: ${SEQ_LENGTH}
  
teacher:
  model_name: ${TEACHER_MODEL}
\`\`\`

## Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "${HF_REPO}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    device_map="auto"
)

# Math problem example
prompt = """Solve the following math problem step by step:

Problem: A train travels from City A to City B at 60 mph and returns at 40 mph. 
What is the average speed for the entire round trip?

Solution:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=512, 
    temperature=0.7,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
\`\`\`

## Training Infrastructure

- **Hardware**: NVIDIA H200/A100 GPUs
- **Container**: \`nvcr.io/nvidia/nemo-rl:v0.4.0\`
- **Checkpointing**: PyTorch DCP format, converted to HuggingFace safetensors

## Limitations

- Optimized specifically for mathematical reasoning tasks
- May not perform optimally on other domains without fine-tuning
- Best used with chain-of-thought prompting

## License

Apache 2.0

## Citation

\`\`\`bibtex
@misc{${MODEL_NAME//-/_},
  author = {DeepBrainz AI},
  title = {${MODEL_NAME}: Math Reasoning via On-Policy Distillation},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/${HF_REPO}}
}
\`\`\`

## Acknowledgments

- [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL) for the training framework
- [Qwen Team](https://huggingface.co/Qwen) for the base models
- [NVIDIA](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) for the training dataset
EOF

# Save training metadata
cat > "${CONVERTED_DIR}/training_args.json" << EOF
{
  "training_method": "on_policy_distillation",
  "teacher_model": "${TEACHER_MODEL}",
  "student_model": "${STUDENT_MODEL}",
  "framework": "nemo-rl",
  "framework_version": "0.4.0",
  "checkpoint_step": ${STEP_NUM},
  "max_sequence_length": ${SEQ_LENGTH},
  "kl_type": "reverse",
  "upload_timestamp": "$(date -Iseconds)"
}
EOF

log "Uploading to HuggingFace..."

# Install huggingface_hub if not available
pip install -q huggingface_hub 2>/dev/null || true

# Login to HuggingFace
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
fi

# Create repo if it doesn't exist
python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo('${HF_REPO}', exist_ok=True, repo_type='model')
    print('Repository created/verified: ${HF_REPO}')
except Exception as e:
    print(f'Note: {e}')
"

# Upload to HuggingFace
huggingface-cli upload "${HF_REPO}" "${CONVERTED_DIR}" . \
    --commit-message "Upload ${MODEL_NAME} - OPD trained at step ${STEP_NUM}"

log "============================================"
log "Upload complete!"
log "Model available at: https://huggingface.co/${HF_REPO}"
log "============================================"

# Cleanup
rm -rf "${CONVERTED_DIR}"

exit 0
