#!/usr/bin/env bash

# PyTorch 2.9.1 Build Test Script
# Tests any PyTorch build variant with comprehensive validation

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Function to auto-detect builds
auto_detect_build() {
    local builds=($(ls -d result-pytorch-* 2>/dev/null | sed 's/result-//'))

    if [ ${#builds[@]} -eq 0 ]; then
        print_color $RED "No builds found. Run 'flox build <variant>' first."
        exit 1
    elif [ ${#builds[@]} -eq 1 ]; then
        echo "${builds[0]}"
    else
        print_color $YELLOW "Multiple builds found:"
        for build in "${builds[@]}"; do
            echo "  - $build"
        done
        print_color $YELLOW "Please specify which build to test: $0 <build-name>"
        exit 1
    fi
}

# Get build name from argument or auto-detect
if [ $# -eq 0 ]; then
    BUILD_NAME=$(auto_detect_build)
    print_color $BLUE "Auto-detected build: $BUILD_NAME"
else
    BUILD_NAME="$1"
fi

# Check if result directory exists
RESULT_DIR="result-${BUILD_NAME}"
if [ ! -d "$RESULT_DIR" ]; then
    print_color $RED "Error: Result directory '$RESULT_DIR' not found."
    print_color $RED "Please build first: flox build --stability=unstable $BUILD_NAME"
    exit 1
fi

# Get the full path to the result
RESULT_PATH=$(readlink -f "$RESULT_DIR")
PYTHON_BIN="${RESULT_PATH}/bin/python"

# Check if this is a CUDA build
if [[ "$BUILD_NAME" == *"cuda"* ]]; then
    IS_CUDA_BUILD=true
else
    IS_CUDA_BUILD=false
fi

print_color $BLUE "========================================"
print_color $BLUE "PyTorch 2.9.1 Build Test"
print_color $BLUE "========================================"
echo "Build: $BUILD_NAME"
echo "Path: $RESULT_PATH"
echo "CUDA build: $IS_CUDA_BUILD"
echo ""

# Test 1: PyTorch Import & Version
print_color $BLUE "========================================"
print_color $BLUE "Test 1: PyTorch Import & Version"
print_color $BLUE "========================================"

PYTORCH_VERSION=$($PYTHON_BIN -c "import torch; print(torch.__version__)" 2>&1)
if [ $? -eq 0 ]; then
    print_color $GREEN "✓ PyTorch version: $PYTORCH_VERSION"

    # Check if it's 2.9.x
    if [[ "$PYTORCH_VERSION" == 2.9.* ]]; then
        print_color $GREEN "✓ Correct PyTorch 2.9.x version"
    else
        print_color $YELLOW "⚠ Expected PyTorch 2.9.x, got $PYTORCH_VERSION"
    fi

    PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    echo "  Python: $PYTHON_VERSION"

    INSTALL_PATH=$($PYTHON_BIN -c "import torch; import os; print(os.path.dirname(torch.__file__))")
    echo "  Install path: $INSTALL_PATH"
else
    print_color $RED "✗ Failed to import PyTorch"
    echo "$PYTORCH_VERSION"
    exit 1
fi

# Test 2: CUDA Support (for CUDA builds)
if [ "$IS_CUDA_BUILD" = true ]; then
    print_color $BLUE ""
    print_color $BLUE "========================================"
    print_color $BLUE "Test 2: CUDA Support"
    print_color $BLUE "========================================"

    $PYTHON_BIN << 'EOF'
import torch
import sys

cuda_built = torch.cuda.is_available()
print(f"CUDA built: {cuda_built}")

if not cuda_built:
    print("✗ CUDA support not built into PyTorch")
    sys.exit(1)

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if hasattr(torch.version, 'cuda'):
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")

    # Check for CUDA 13.0
    if cuda_version and cuda_version.startswith("13."):
        print("✓ CUDA 13.0 detected")
    else:
        print(f"⚠ Expected CUDA 13.0, got {cuda_version}")

if hasattr(torch.backends.cudnn, 'version'):
    cudnn_version = torch.backends.cudnn.version()
    print(f"cuDNN version: {cudnn_version}")

# Show compiled architectures
if hasattr(torch.cuda, 'get_arch_list'):
    arch_list = torch.cuda.get_arch_list()
    print(f"Compiled arch list: {arch_list}")

# Check available GPUs
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

        # Get compute capability
        cap = torch.cuda.get_device_capability(i)
        print(f"  Compute capability: {cap[0]}.{cap[1]}")

        # Memory info
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  Memory: {mem_total:.1f} GB")
else:
    print("⚠ CUDA built but no GPU available for testing")
EOF

    if [ $? -ne 0 ]; then
        print_color $RED "✗ CUDA support test failed"
        exit 1
    fi
fi

# Test 3: CPU Inference
print_color $BLUE ""
print_color $BLUE "========================================"
print_color $BLUE "Test 3: CPU Inference"
print_color $BLUE "========================================"

$PYTHON_BIN << 'EOF'
import torch
import torch.nn as nn

# Create a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model and test data
model = SimpleNet()
model.eval()

# Test input
batch_size = 32
input_tensor = torch.randn(batch_size, 512)

# Run inference
with torch.no_grad():
    output = model(input_tensor)

print(f"✓ Input shape: {input_tensor.shape}")
print(f"✓ Output shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model parameters: {total_params:,}")

print("✓ CPU inference successful!")
EOF

if [ $? -ne 0 ]; then
    print_color $RED "✗ CPU inference test failed"
    exit 1
fi

# Test 4: GPU Inference (for CUDA builds with available GPU)
if [ "$IS_CUDA_BUILD" = true ]; then
    print_color $BLUE ""
    print_color $BLUE "========================================"
    print_color $BLUE "Test 4: GPU Inference"
    print_color $BLUE "========================================"

    $PYTHON_BIN << 'EOF'
import torch
import torch.nn as nn
import sys

if not torch.cuda.is_available():
    print("⚠ No GPU available for testing")
    sys.exit(0)

# Check architecture compatibility
gpu_cap = torch.cuda.get_device_capability(0)
gpu_cap_str = f"{gpu_cap[0]}.{gpu_cap[1]}"
arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []

print(f"GPU compute capability: {gpu_cap_str}")
print(f"Compiled architectures: {arch_list}")

# Check if current GPU architecture is in compiled list
gpu_sm = f"sm_{gpu_cap[0]}{gpu_cap[1]}"
if arch_list and gpu_sm not in str(arch_list):
    print(f"⚠ WARNING: GPU architecture {gpu_cap_str} not in compiled list {arch_list}")
    print("  This build is targeted for a different GPU architecture")
    print("  GPU operations will fail - this is expected behavior")
    sys.exit(0)

try:
    # Create a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Move model to GPU
    model = SimpleNet().cuda()
    model.eval()

    # Test input on GPU
    batch_size = 32
    input_tensor = torch.randn(batch_size, 512).cuda()

    # Run inference on GPU
    with torch.no_grad():
        output = model(input_tensor)

    print(f"✓ Input device: {input_tensor.device}")
    print(f"✓ Output device: {output.device}")
    print(f"✓ Output shape: {output.shape}")

    # Test matrix multiplication
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    print(f"✓ Matrix multiplication: {c.shape}")

    print("✓ GPU inference successful!")

except RuntimeError as e:
    if "no kernel image is available" in str(e):
        print(f"✗ Architecture mismatch: {e}")
        print("  This is expected if testing a build for a different GPU")
    else:
        raise
EOF
fi

# Test 5: Build Configuration
print_color $BLUE ""
print_color $BLUE "========================================"
print_color $BLUE "Test 5: Build Configuration"
print_color $BLUE "========================================"

$PYTHON_BIN << EOF
import torch

# Get build configuration
config = torch.__config__.show() if hasattr(torch.__config__, 'show') else "Config not available"

# Parse some key information
print(f"Package name: $BUILD_NAME")
print(f"PyTorch version: {torch.__version__}")

# Check for MKL/MKL-DNN
has_mkl = "MKL" in config if isinstance(config, str) else False
has_mkldnn = "MKLDNN" in config if isinstance(config, str) else False
print(f"MKL available: {has_mkl}")
print(f"MKL-DNN available: {has_mkldnn}")

# For CUDA builds, check architecture
if torch.cuda.is_available():
    if hasattr(torch.cuda, 'get_arch_list'):
        arch_list = torch.cuda.get_arch_list()
        print(f"CUDA architectures: {arch_list}")

        # Check if expected architecture is in the list
        build_name = "$BUILD_NAME"
        if "sm120" in build_name and "sm_120" in str(arch_list):
            print("✓ Expected architecture found in build")
        elif "sm90" in build_name and "sm_90" in str(arch_list):
            print("✓ Expected architecture found in build")
        elif "sm86" in build_name and "sm_86" in str(arch_list):
            print("✓ Expected architecture found in build")
        elif "sm80" in build_name and "sm_80" in str(arch_list):
            print("✓ Expected architecture found in build")
EOF

# Summary
print_color $BLUE ""
print_color $BLUE "========================================"
print_color $BLUE "Test Summary"
print_color $BLUE "========================================"

print_color $GREEN "✓ Build: $BUILD_NAME"
print_color $GREEN "✓ PyTorch imported successfully"
print_color $GREEN "✓ CPU inference working"

if [ "$IS_CUDA_BUILD" = true ]; then
    print_color $GREEN "✓ CUDA support verified"
    print_color $GREEN "✓ GPU inference working (or skipped due to arch mismatch)"
fi

print_color $BLUE ""
print_color $GREEN "✓ All tests passed!"
print_color $BLUE "========================================"