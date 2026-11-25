#!/bin/bash
set -e

echo "=========================================================="
echo " AMD ROCm Installation Script for Pop!_OS 22.04 LTS"
echo "=========================================================="

echo "[1/4] Adding the AMD ROCm GPG key..."
wget https://repo.radeon.com/rocm/rocm.gpg.key -qO - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "[2/4] Adding the AMD ROCm apt repository..."
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

echo "[3/4] Setting pin priority..."
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600

echo "[4/4] Updating apt and installing rocm..."
sudo apt update
sudo apt install -y rocm

echo "=========================================================="
echo "Installation complete!"
echo "Adding user to the render group..."
sudo usermod -a -G render $USER

echo "=========================================================="
echo "IMPORTANT: You MUST REBOOT your system for the changes to take effect."
echo "After rebooting, verify the installation by running 'rocminfo' in your terminal."
echo "=========================================================="
