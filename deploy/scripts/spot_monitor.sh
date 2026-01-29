#!/bin/bash
# spot_monitor.sh - EC2 Spot Instance Termination Monitor
# 
# This script monitors the EC2 instance metadata endpoint for spot termination notices.
# When a termination is detected, it attempts to gracefully stop the training container.
#
# Usage: ./spot_monitor.sh [container_name]
#
# Note: NeMo-RL saves checkpoints periodically based on save_period config.
# This script provides a best-effort graceful shutdown but does NOT guarantee
# a final checkpoint save (NeMo-RL doesn't have built-in SIGUSR1 handling for this).

set -uo pipefail

CONTAINER_NAME="${1:-nemo-opd}"
METADATA_URL="http://169.254.169.254/latest/meta-data/spot/termination-time"
CHECK_INTERVAL="${CHECK_INTERVAL:-5}"
GRACE_PERIOD="${GRACE_PERIOD:-90}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SPOT-MONITOR] $*"
}

log "============================================"
log "EC2 Spot Termination Monitor Started"
log "============================================"
log "Container Name: ${CONTAINER_NAME}"
log "Check Interval: ${CHECK_INTERVAL}s"
log "Grace Period:   ${GRACE_PERIOD}s"
log "============================================"

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"
}

# Function to handle termination
handle_termination() {
    log "============================================"
    log "SPOT TERMINATION NOTICE DETECTED!"
    log "============================================"
    
    # Log termination time
    TERM_TIME=$(curl -s --connect-timeout 2 "${METADATA_URL}" 2>/dev/null || echo "unknown")
    log "Termination scheduled at: ${TERM_TIME}"
    
    # Check if container is running
    if container_running; then
        log "Container ${CONTAINER_NAME} is running"
        
        # Try to send SIGTERM to python processes (graceful shutdown)
        log "Sending SIGTERM to training process..."
        docker exec "${CONTAINER_NAME}" pkill -TERM -f "python.*distillation" 2>/dev/null || true
        
        # Wait for grace period to allow checkpoint save
        log "Waiting ${GRACE_PERIOD}s for potential checkpoint save..."
        sleep "${GRACE_PERIOD}"
        
        # Stop container gracefully
        log "Stopping container gracefully..."
        docker stop -t 30 "${CONTAINER_NAME}" 2>/dev/null || true
        
        log "Container stopped"
    else
        log "Container ${CONTAINER_NAME} is not running"
    fi
    
    log "Spot monitor shutdown complete"
    exit 0
}

# Main monitoring loop
log "Starting monitoring loop..."

while true; do
    # Check spot termination metadata
    RESPONSE=$(curl -s --connect-timeout 2 "${METADATA_URL}" 2>/dev/null || echo "")
    
    # If response contains a timestamp (ISO 8601 format), termination is scheduled
    if echo "${RESPONSE}" | grep -qE "^[0-9]{4}-[0-9]{2}-[0-9]{2}T"; then
        handle_termination
    fi
    
    # Also check container health
    if ! container_running; then
        log "WARNING: Container ${CONTAINER_NAME} is not running"
    fi
    
    sleep "${CHECK_INTERVAL}"
done
