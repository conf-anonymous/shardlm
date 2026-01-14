#!/bin/bash
# ShardLM Example Runner Script
# Runs examples for V2, V3, and V3 variants (MPC, OT, CC)
#
# Usage:
#   ./scripts/run_examples.sh           # Run all examples
#   ./scripts/run_examples.sh --v2      # Run V2 examples only
#   ./scripts/run_examples.sh --v3      # Run V3 examples only
#   ./scripts/run_examples.sh --v3-mpc  # Run V3-MPC examples only
#   ./scripts/run_examples.sh --v3-ot   # Run V3-OT examples only
#   ./scripts/run_examples.sh --v3-cc   # Run V3-CC examples only

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

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

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Default settings
MODEL_DIR="${SHARDLM_V2_MODEL_DIR:-/workspace/qwen2.5-1.5b-instruct-weights}"
MODEL_ARCH="${SHARDLM_V2_MODEL_ARCH:-qwen2_5_1_5b}"
SERVER_URL="${SHARDLM_V2_SERVER_URL:-http://localhost:9090}"
NUM_GPUS="${SHARDLM_V2_NUM_GPUS:-1}"
TEST_PROMPT="What is 2 + 2?"
MAX_TOKENS=20

# Parse command line arguments
RUN_V2=false
RUN_V3=false
RUN_V3_MPC=false
RUN_V3_OT=false
RUN_V3_CC=false
RUN_ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --v2)
            RUN_V2=true
            RUN_ALL=false
            shift
            ;;
        --v3)
            RUN_V3=true
            RUN_ALL=false
            shift
            ;;
        --v3-mpc)
            RUN_V3_MPC=true
            RUN_ALL=false
            shift
            ;;
        --v3-ot)
            RUN_V3_OT=true
            RUN_ALL=false
            shift
            ;;
        --v3-cc)
            RUN_V3_CC=true
            RUN_ALL=false
            shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --v2          Run V2 examples only"
            echo "  --v3          Run V3 examples only"
            echo "  --v3-mpc      Run V3-MPC examples only"
            echo "  --v3-ot       Run V3-OT examples only"
            echo "  --v3-cc       Run V3-CC examples only"
            echo "  --model-dir   Path to model weights"
            echo "  --server-url  Server URL (default: http://localhost:9090)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$RUN_ALL" = true ]; then
    RUN_V2=true
    RUN_V3=true
    RUN_V3_MPC=true
    RUN_V3_OT=true
    RUN_V3_CC=true
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_header "ShardLM Example Runner"

# Check prerequisites
print_step "Checking prerequisites..."

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Is this a GPU machine?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
print_info "GPU: $GPU_NAME"

if [ ! -f "target/release/shardlm-v2-client" ]; then
    print_warn "Client not built. Building now..."
    cargo build -p shardlm-v2-client --release
fi

# Check if server is running
check_server() {
    if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start server with specific features
start_server() {
    local features="$1"
    local server_name="$2"

    print_step "Building server with features: $features"
    cargo build -p shardlm-v2-server --features "$features" --release 2>&1 | tail -3

    print_step "Starting $server_name server..."
    SHARDLM_V2_MODEL_DIR="$MODEL_DIR" \
    SHARDLM_V2_MODEL_ARCH="$MODEL_ARCH" \
    SHARDLM_V2_NUM_GPUS="$NUM_GPUS" \
    SHARDLM_V2_PORT=9090 \
    RUST_LOG=info \
    ./target/release/shardlm-v2-server &
    SERVER_PID=$!

    # Wait for server to be ready (up to 3 minutes for large models)
    print_info "Waiting for server to start (may take 2-3 minutes for weight loading)..."
    for i in {1..180}; do
        if check_server; then
            print_success "Server started (PID: $SERVER_PID)"
            return 0
        fi
        # Show progress every 30 seconds
        if [ $((i % 30)) -eq 0 ]; then
            print_info "Still waiting... ($i seconds)"
        fi
        sleep 1
    done

    print_error "Server failed to start within 180 seconds"
    kill $SERVER_PID 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        print_info "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        unset SERVER_PID
    fi
}

# Cleanup on exit
trap stop_server EXIT

# Run example for a specific endpoint
run_example() {
    local endpoint="$1"
    local description="$2"

    print_step "Testing $description..."
    echo "  Prompt: \"$TEST_PROMPT\""
    echo "  Endpoint: $endpoint"
    echo ""

    if ./target/release/shardlm-v2-client generate \
        -s "$SERVER_URL" \
        -p "$TEST_PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --endpoint "$endpoint" \
        --timing \
        --tokenizer "$MODEL_DIR/tokenizer.json" 2>&1; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

RESULTS=()

# V2 Examples
if [ "$RUN_V2" = true ]; then
    print_header "V2 Examples (Base GPU Implementation)"

    if ! check_server; then
        start_server "cuda" "V2/V3"
    fi

    if run_example "v2" "V2 Secure Inference"; then
        RESULTS+=("V2: PASS")
    else
        RESULTS+=("V2: FAIL")
    fi
fi

# V3 Examples (baseline)
if [ "$RUN_V3" = true ]; then
    print_header "V3 Examples (Baseline - uses _approx functions)"

    if ! check_server; then
        start_server "cuda" "V3"
    fi

    if run_example "v3" "V3 Secure Inference (Baseline)"; then
        RESULTS+=("V3: PASS")
    else
        RESULTS+=("V3: FAIL")
    fi
fi

# V3-OT Examples
if [ "$RUN_V3_OT" = true ]; then
    print_header "V3-OT Examples (Oblivious Transfer)"

    stop_server

    if ! check_server; then
        start_server "cuda" "V3-OT"
    fi

    # Test OT info endpoint first
    print_step "Testing V3-OT info endpoint..."
    if curl -s "$SERVER_URL/v3/ot/info" | head -c 200; then
        echo ""
        print_success "OT info endpoint working"
    else
        print_warn "OT info endpoint not available"
    fi

    if run_example "v3-ot" "V3-OT Secure Inference"; then
        RESULTS+=("V3-OT: PASS")
    else
        RESULTS+=("V3-OT: FAIL")
    fi
fi

# V3-MPC Examples
if [ "$RUN_V3_MPC" = true ]; then
    print_header "V3-MPC Examples (Beaver Triples)"

    stop_server

    print_step "Building server with MPC support..."
    cargo build -p shardlm-v2-server --features "cuda,mpc-secure" --release 2>&1 | tail -3

    if ! check_server; then
        start_server "cuda,mpc-secure" "V3-MPC"
    fi

    # Test MPC info endpoint first
    print_step "Testing V3-MPC info endpoint..."
    if curl -s "$SERVER_URL/v3/mpc/info" | head -c 200; then
        echo ""
        print_success "MPC info endpoint working"
    else
        print_warn "MPC info endpoint not available (feature may not be enabled)"
    fi

    if run_example "v3-mpc" "V3-MPC Secure Inference"; then
        RESULTS+=("V3-MPC: PASS")
    else
        RESULTS+=("V3-MPC: FAIL")
    fi
fi

# V3-CC Examples
if [ "$RUN_V3_CC" = true ]; then
    print_header "V3-CC Examples (H100 Confidential Computing)"

    stop_server

    print_step "Building server with CC support..."
    cargo build -p shardlm-v2-server --features "h100-cc,cuda" --release 2>&1 | tail -3

    if ! check_server; then
        start_server "h100-cc,cuda" "V3-CC"
    fi

    # Test CC attestation endpoint first
    print_step "Testing V3-CC attestation endpoint..."
    if curl -s "$SERVER_URL/v3/cc/attestation" | head -c 200; then
        echo ""
        print_success "CC attestation endpoint working"
    else
        print_warn "CC attestation endpoint not available (H100 may not be present)"
    fi

    if run_example "v3-cc" "V3-CC Secure Inference"; then
        RESULTS+=("V3-CC: PASS")
    else
        RESULTS+=("V3-CC: FAIL")
    fi
fi

# Summary
print_header "Summary"

echo "Results:"
for result in "${RESULTS[@]}"; do
    if [[ "$result" == *"PASS"* ]]; then
        echo -e "  ${GREEN}✓${NC} $result"
    else
        echo -e "  ${RED}✗${NC} $result"
    fi
done

echo ""
print_success "Example run complete!"
