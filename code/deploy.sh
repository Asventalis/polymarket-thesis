#!/bin/bash

# Configuration
REMOTE_USER="v.pampinella"
REMOTE_HOST="131.114.142.5"
REMOTE_DIR="/data/v.pampinella/agy_project"

# Ensure the remote directory exists (retry until it succeeds)
while ! ssh -o ConnectTimeout=10 -o ServerAliveInterval=60 $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"; do
    echo "SSH connection failed. Retrying in 10 seconds..."
    sleep 10
done

# Rsync the codebase and data, carefully excluding local virtual environments and caches
# Wrap in an infinite while loop to auto-resume on drop
while ! rsync -avz --progress --partial -e "ssh -o ServerAliveInterval=60" \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude '__pycache__' \
    --exclude '.tmp' \
    --exclude '*.duckdb' \
    --exclude 'multi_day' \
    ./ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/; do
    
    echo "Rsync transfer interrupted! The VPN might be disconnected. Retrying in 10 seconds..."
    sleep 10
done

echo "Sync complete! Project is now safely on the remote server in $REMOTE_DIR"
