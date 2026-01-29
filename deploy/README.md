# NeMo-RL Training Deployment

Production-ready On-Policy Distillation (OPD) and GRPO training deployment for AWS EC2 instances.

**Based on actual tested NeMo-RL v0.4.0 configurations.**

## Instance Configurations (ALL UNIQUE)

| Instance | Algorithm | Teacher/Model | Student | HuggingFace Repo |
|----------|-----------|---------------|---------|------------------|
| `h200-1` | **OPD** | Qwen3-32B | Qwen3-4B-Base | `arunkumarvr/deepbrainz-r1-4b-math-reasoning` |
| `h200-2` | **OPD** | Qwen3-32B | Qwen3-1.7B-Base | `arunkumarvr/deepbrainz-r1-1.7b-math-reasoning` |
| `a100-80` | **OPD** | Qwen3-4B | Qwen3-1.7B-Base | `arunkumarvr/deepbrainz-r1-1.7b-math-4b-teacher` |
| `a100-40` | **GRPO** | Qwen3-8B-Base | - | `arunkumarvr/deepbrainz-r1-8b-math-grpo` |

### Config Sources (from NeMo-RL repo)

| Instance | NeMo-RL Config File |
|----------|---------------------|
| `h200-1` | `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-1n8g-fsdp2tp2-dynamicbatch.v1.yaml` |
| `h200-2` | `examples/configs/recipes/llm/distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1.yaml` |
| `a100-80` | `examples/configs/distillation_math.yaml` (default tested config) |
| `a100-40` | `examples/configs/recipes/llm/grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.yaml` |

## Quick Start

### Step 1: Connect to instance via SSM

```bash
aws ssm start-session --target <INSTANCE-ID>
sudo su - ubuntu
```

### Step 2: Download and run setup script

```bash
# Download the setup script
curl -LO https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main/deploy/setup_training.sh
chmod +x setup_training.sh

# Run setup with your credentials
./setup_training.sh <instance-type> <wandb-api-key> <hf-token>
```

### Examples

```bash
# H200 Instance #1 - OPD: 32B → 4B
./setup_training.sh h200-1 wandb_xxx hf_xxx

# H200 Instance #2 - OPD: 32B → 1.7B  
./setup_training.sh h200-2 wandb_xxx hf_xxx

# A100-80GB - OPD: 4B → 1.7B
./setup_training.sh a100-80 wandb_xxx hf_xxx

# A100-40GB - GRPO: 8B-Base
./setup_training.sh a100-40 wandb_xxx hf_xxx
```

### Step 3: Monitor training

```bash
# View training logs
sudo journalctl -u nemo-training -f

# Check WandB dashboard
# OPD: https://wandb.ai/YOUR_ORG/deepbrainz-distillation
# GRPO: https://wandb.ai/YOUR_ORG/deepbrainz-grpo

# Check checkpoint status
ls -la /mnt/ckpt/<job-name>/
```

## Directory Structure

```
deploy/
├── setup_training.sh        # Master setup script (ALL-IN-ONE)
├── README.md                # This file
├── scripts/
│   ├── run_opd.sh           # OPD training wrapper (reference)
│   ├── upload_to_hf.sh      # HuggingFace upload script (reference)
│   └── spot_monitor.sh      # Spot termination handler (reference)
├── configs/
│   ├── h200-1.env           # H200 #1: OPD 32B → 4B-Base
│   ├── h200-2.env           # H200 #2: OPD 32B → 1.7B-Base
│   ├── a100-80.env          # A100-80: OPD 4B → 1.7B-Base
│   └── a100-40.env          # A100-40: GRPO 8B-Base
└── systemd/
    ├── nemo-opd.service     # Training service template
    └── spot-monitor.service # Spot monitor service
```

## Commands Reference

```bash
# Start training
sudo systemctl start nemo-training

# Stop training
sudo systemctl stop nemo-training

# Restart training  
sudo systemctl restart nemo-training

# View logs (live)
sudo journalctl -u nemo-training -f

# Check status
sudo systemctl status nemo-training

# View container logs directly
sudo docker logs -f nemo-distillation  # or nemo-grpo

# Enter container for debugging
sudo docker exec -it nemo-distillation bash  # or nemo-grpo

# Check GPU usage
nvidia-smi -l 5

# List checkpoints
ls -la /mnt/ckpt/<job-name>/
```

## What Happens on Spot Termination

1. `spot_monitor.sh` detects termination notice (2-minute warning)
2. Sends SIGTERM to training process
3. Waits 90 seconds for potential checkpoint
4. Gracefully stops container
5. Instance terminates

**Important**: Checkpoints are saved to `/mnt/ckpt/` (EBS) every 25 steps.

## Expected Outputs After 48 Hours

| Model | Training | HuggingFace |
|-------|----------|-------------|
| Qwen3-4B (distilled from 32B) | H200 #1 | `deepbrainz-r1-4b-math-reasoning` |
| Qwen3-1.7B (distilled from 32B) | H200 #2 | `deepbrainz-r1-1.7b-math-reasoning` |
| Qwen3-1.7B (distilled from 4B) | A100-80 | `deepbrainz-r1-1.7b-math-4b-teacher` |
| Qwen3-8B (GRPO trained) | A100-40 | `deepbrainz-r1-8b-math-grpo` |

## Container Information

- **Image**: `nvcr.io/nvidia/nemo-rl:v0.4.0`
- **Working Directory**: `/opt/nemo-rl`
- **OPD Entry Point**: `examples/run_distillation_math.py`
- **GRPO Entry Point**: `examples/run_grpo_math.py`

## Troubleshooting

### Training won't start

```bash
# Check container image
sudo docker images | grep nemo-rl

# Pull image manually
sudo docker pull nvcr.io/nvidia/nemo-rl:v0.4.0

# Check service status
sudo systemctl status nemo-training
sudo journalctl -u nemo-training --no-pager -n 100
```

### Out of memory

```bash
# Edit service file to add memory overrides
sudo nano /etc/systemd/system/nemo-training.service

# Add to EXTRA_OVERRIDES:
# policy.generation.vllm_cfg.gpu_memory_utilization=0.3

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart nemo-training
```

## License

Apache 2.0
