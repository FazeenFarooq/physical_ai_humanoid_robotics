#!/bin/bash
# Setup script for Physical AI & Humanoid Robotics Course Development Environment
# This script helps configure the development environment with Python 3.10, CUDA 12.2, and PyTorch 2.0+

echo "Setting up development environment for Physical AI & Humanoid Robotics Course..."

# Check if running on Windows Subsystem for Linux (WSL) or native Linux
if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
    echo "Detected WSL - please ensure you have proper GPU passthrough configured for CUDA"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
if (( $(echo "$PYTHON_VERSION < 3.10" | bc -l) )); then
    echo "Python 3.10 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
else
    echo "Python version $PYTHON_VERSION is sufficient"
fi

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip3 install --user -r requirements.txt

# Verify CUDA availability (if NVIDIA GPU is present)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Checking CUDA version..."
    nvidia-smi
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+')
    if [ -z "$CUDA_VERSION" ]; then
        echo "CUDA toolkit not found. Please install CUDA 12.2 or higher."
        exit 1
    else
        if (( $(echo "$CUDA_VERSION < 12.2" | bc -l) )); then
            echo "CUDA 12.2 or higher is required. Current version: $CUDA_VERSION"
            exit 1
        else
            echo "CUDA version $CUDA_VERSION is sufficient"
        fi
    fi
else
    echo "Warning: No nvidia-smi found. Make sure you have an NVIDIA GPU with proper drivers installed."
fi

# Verify PyTorch with CUDA availability
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')" || echo "PyTorch installation failed"

# Verify other important dependencies
echo "Verifying important dependencies..."
python3 -c "import rclpy; print('ROS 2 Python client available')" || echo "ROS 2 Python client not available - ensure ROS 2 Humble is installed"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Development environment setup completed!"
echo "Remember to source ROS 2 before running ROS 2 commands: source /opt/ros/humble/setup.bash"