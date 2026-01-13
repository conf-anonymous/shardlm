#!/bin/bash
# ShardLM Setup Script for NVIDIA GPU Machines
# Supports RunPod, Lambda Labs, and other cloud GPU providers
#
# Usage:
#   ./scripts/setup_nvidia_machine.sh           # Interactive model selection
#   ./scripts/setup_nvidia_machine.sh --1.5b    # Install 1.5B model only
#   ./scripts/setup_nvidia_machine.sh --7b      # Install 7B model only
#   ./scripts/setup_nvidia_machine.sh --all     # Install both models
#   ./scripts/setup_nvidia_machine.sh --skip    # Skip model download

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo ""
echo "========================================"
echo "  ShardLM Setup for NVIDIA GPU Machines"
echo "========================================"
echo ""

# Parse command line arguments
MODEL_CHOICE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --1.5b|--1.5B)
            MODEL_CHOICE="1"
            shift
            ;;
        --7b|--7B)
            MODEL_CHOICE="2"
            shift
            ;;
        --all)
            MODEL_CHOICE="3"
            shift
            ;;
        --skip)
            MODEL_CHOICE="4"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Usage: $0 [--1.5b|--7b|--all|--skip]"
            exit 1
            ;;
    esac
done

# Step 1: Check GPU
print_step "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME"
    echo "  VRAM: $GPU_MEMORY"

    # Check for specific GPU types
    if echo "$GPU_NAME" | grep -qi "H100"; then
        print_info "H100 detected - all features available including V3-CC"
    elif echo "$GPU_NAME" | grep -qi "H200"; then
        print_info "H200 detected - all features available including V3-CC"
    elif echo "$GPU_NAME" | grep -qi "A100\|A40\|RTX 4090\|RTX 3090"; then
        print_info "High-end GPU detected - V2, V3, V3-MPC, V3-OT available"
    else
        print_warn "GPU may have limited VRAM. Check memory requirements."
    fi
else
    print_error "nvidia-smi not found. Is this a GPU machine?"
    exit 1
fi
echo ""

# Step 2: Install Rust
print_step "Checking Rust installation..."
if command -v cargo &> /dev/null; then
    echo "  Rust already installed: $(rustc --version)"
else
    print_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "  Rust installed: $(rustc --version)"
fi
echo ""

# Step 3: Install Python dependencies
print_step "Checking Python and huggingface_hub..."
if command -v python3 &> /dev/null; then
    echo "  Python: $(python3 --version)"
    pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub 2>/dev/null
    echo "  huggingface_hub installed"
else
    print_warn "Python3 not found. Model download may fail."
fi
echo ""

# Step 4: Setup workspace
print_step "Setting up workspace..."
WORKSPACE_DIR="/workspace/shardlm"

if [ -d "$WORKSPACE_DIR" ]; then
    echo "  Workspace exists at $WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
else
    print_info "Cloning repository..."
    cd /workspace
    # Clone from anonymous repo (update URL as needed)
    git clone https://github.com/anonymous/shardlm.git shardlm 2>/dev/null || {
        print_error "Failed to clone repository. Please clone manually:"
        echo "  cd /workspace && git clone <repo-url> shardlm"
        exit 1
    }
    cd "$WORKSPACE_DIR"
fi
echo ""

# Step 5: Build server and client
print_step "Building ShardLM (this may take 5-10 minutes on first run)..."
echo "  Building server with CUDA support..."
cargo build -p shardlm-v2-server --features cuda --release 2>&1 | tail -3
echo "  Building client..."
cargo build -p shardlm-v2-client --release 2>&1 | tail -3
echo "  Build complete!"
echo ""

# Step 6: Model selection
print_step "Model selection..."

if [ -z "$MODEL_CHOICE" ]; then
    echo ""
    echo "  Select model to download:"
    echo "    1) Qwen 2.5 1.5B  (~3 GB,  requires ~8 GB VRAM)"
    echo "    2) Qwen 2.5 7B    (~14 GB, requires ~20 GB VRAM)"
    echo "    3) Both models"
    echo "    4) Skip model download"
    echo ""
    read -p "  Enter choice [1-4]: " MODEL_CHOICE
fi

MODEL_1_5B_DIR="/workspace/qwen2.5-1.5b-instruct-weights"
MODEL_7B_DIR="/workspace/qwen2.5-7b-instruct-weights"

download_1_5b() {
    if [ -d "$MODEL_1_5B_DIR" ] && [ -f "$MODEL_1_5B_DIR/config.json" ]; then
        print_info "Qwen 2.5 1.5B already exists at $MODEL_1_5B_DIR"
    else
        print_info "Downloading Qwen 2.5 1.5B (~3 GB)..."
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$MODEL_1_5B_DIR')"
        echo "  Downloaded to $MODEL_1_5B_DIR"
    fi
}

download_7b() {
    if [ -d "$MODEL_7B_DIR" ] && [ -f "$MODEL_7B_DIR/config.json" ]; then
        print_info "Qwen 2.5 7B already exists at $MODEL_7B_DIR"
    else
        print_info "Downloading Qwen 2.5 7B (~14 GB)..."
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$MODEL_7B_DIR')"
        echo "  Downloaded to $MODEL_7B_DIR"
    fi
}

case $MODEL_CHOICE in
    1)
        download_1_5b
        DEFAULT_MODEL_DIR="$MODEL_1_5B_DIR"
        DEFAULT_MODEL_ARCH="qwen2_5_1_5b"
        ;;
    2)
        download_7b
        DEFAULT_MODEL_DIR="$MODEL_7B_DIR"
        DEFAULT_MODEL_ARCH="qwen2_5_7b"
        ;;
    3)
        download_1_5b
        download_7b
        DEFAULT_MODEL_DIR="$MODEL_1_5B_DIR"
        DEFAULT_MODEL_ARCH="qwen2_5_1_5b"
        ;;
    4)
        print_info "Skipping model download"
        DEFAULT_MODEL_DIR="/workspace/model-weights"
        DEFAULT_MODEL_ARCH="qwen2_5_1_5b"
        ;;
    *)
        print_error "Invalid choice. Skipping model download."
        DEFAULT_MODEL_DIR="/workspace/model-weights"
        DEFAULT_MODEL_ARCH="qwen2_5_1_5b"
        ;;
esac
echo ""

# Step 7: Summary
print_step "Setup Complete!"
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo "  GPU:           $GPU_NAME ($GPU_MEMORY)"
echo "  Workspace:     $WORKSPACE_DIR"
echo "  Server:        $WORKSPACE_DIR/target/release/shardlm-v2-server"
echo "  Client:        $WORKSPACE_DIR/target/release/shardlm-v2-client"
if [ "$MODEL_CHOICE" != "4" ]; then
    echo "  Model:         $DEFAULT_MODEL_DIR"
fi
echo ""
echo "========================================"
echo "  Quick Start"
echo "========================================"
echo ""
echo "1. Start the server:"
echo ""
echo "   SHARDLM_V2_MODEL_DIR=$DEFAULT_MODEL_DIR \\"
echo "   SHARDLM_V2_MODEL_ARCH=$DEFAULT_MODEL_ARCH \\"
echo "   SHARDLM_V2_PORT=9090 \\"
echo "   ./target/release/shardlm-v2-server"
echo ""
echo "2. In another terminal, test with:"
echo ""
echo "   ./target/release/shardlm-v2-client health -s http://localhost:9090"
echo "   ./target/release/shardlm-v2-client generate -s http://localhost:9090 -p \"Hello!\""
echo ""
echo "3. Run benchmarks:"
echo ""
echo "   ./target/release/shardlm-v2-client benchmark -s http://localhost:9090 \\"
echo "     --runs 10 --warmup 2 --endpoint v3-ot"
echo ""
echo "========================================"
