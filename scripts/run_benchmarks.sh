#!/bin/bash
# ShardLM Benchmark Runner Script
# Runs benchmarks and saves results to benchmarks_results/
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run all benchmarks
#   ./scripts/run_benchmarks.sh --v2         # Run V2 benchmarks only
#   ./scripts/run_benchmarks.sh --v3         # Run V3 benchmarks only
#   ./scripts/run_benchmarks.sh --v3-mpc     # Run V3-MPC benchmarks only
#   ./scripts/run_benchmarks.sh --v3-ot      # Run V3-OT benchmarks only
#   ./scripts/run_benchmarks.sh --v3-cc      # Run V3-CC benchmarks only
#   ./scripts/run_benchmarks.sh --sharing    # Run sharing/crypto benchmarks

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
BENCHMARK_ITERATIONS="${BENCHMARK_ITERATIONS:-10}"
MAX_TOKENS="${BENCHMARK_MAX_TOKENS:-50}"

# Parse command line arguments
RUN_V2=false
RUN_V3=false
RUN_V3_MPC=false
RUN_V3_OT=false
RUN_V3_CC=false
RUN_SHARING=false
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
        --sharing)
            RUN_SHARING=true
            RUN_ALL=false
            shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --iterations)
            BENCHMARK_ITERATIONS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --v2            Run V2 benchmarks only"
            echo "  --v3            Run V3 benchmarks only"
            echo "  --v3-mpc        Run V3-MPC benchmarks only"
            echo "  --v3-ot         Run V3-OT benchmarks only"
            echo "  --v3-cc         Run V3-CC benchmarks only"
            echo "  --sharing       Run sharing/crypto benchmarks only"
            echo "  --model-dir     Path to model weights"
            echo "  --iterations    Number of benchmark iterations (default: 10)"
            echo "  --max-tokens    Max tokens to generate (default: 50)"
            echo "  --help          Show this help message"
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
    RUN_SHARING=true
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PROJECT_ROOT/benchmarks_results"
RUN_DIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

print_header "ShardLM Benchmark Runner"

print_info "Results will be saved to: $RUN_DIR"

# Check prerequisites
print_step "Checking prerequisites..."

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Is this a GPU machine?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
print_info "GPU: $GPU_NAME ($GPU_MEM)"

# Save system info
{
    echo "ShardLM Benchmark Results"
    echo "========================="
    echo "Timestamp: $(date)"
    echo "GPU: $GPU_NAME"
    echo "GPU Memory: $GPU_MEM"
    echo "Model: $MODEL_ARCH"
    echo "Model Dir: $MODEL_DIR"
    echo "Iterations: $BENCHMARK_ITERATIONS"
    echo "Max Tokens: $MAX_TOKENS"
    echo ""
    nvidia-smi
} > "$RUN_DIR/system_info.txt"

# Build client if needed
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
    RUST_LOG=warn \
    ./target/release/shardlm-v2-server &
    SERVER_PID=$!

    # Wait for server to be ready
    print_info "Waiting for server to start..."
    for i in {1..60}; do
        if check_server; then
            print_success "Server started (PID: $SERVER_PID)"
            return 0
        fi
        sleep 1
    done

    print_error "Server failed to start within 60 seconds"
    kill $SERVER_PID 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        print_info "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        unset SERVER_PID
        sleep 2
    fi
}

# Cleanup on exit
trap stop_server EXIT

# Run inference benchmark for a specific endpoint
run_inference_benchmark() {
    local endpoint="$1"
    local description="$2"
    local output_file="$RUN_DIR/${endpoint}_benchmark.txt"

    print_step "Benchmarking $description..."
    print_info "Output: $output_file"

    {
        echo "Benchmark: $description"
        echo "Endpoint: $endpoint"
        echo "Iterations: $BENCHMARK_ITERATIONS"
        echo "Max Tokens: $MAX_TOKENS"
        echo "Timestamp: $(date)"
        echo ""
        echo "Results:"
        echo "--------"
    } > "$output_file"

    # Run benchmark with client
    if ./target/release/shardlm-v2-client benchmark \
        -s "$SERVER_URL" \
        --endpoint "$endpoint" \
        --iterations "$BENCHMARK_ITERATIONS" \
        --max-tokens "$MAX_TOKENS" \
        --tokenizer "$MODEL_DIR/tokenizer.json" 2>&1 | tee -a "$output_file"; then
        print_success "$description benchmark completed"
        return 0
    else
        print_error "$description benchmark failed"
        echo "FAILED" >> "$output_file"
        return 1
    fi
}

# Run cargo benchmarks
run_cargo_benchmarks() {
    local package="$1"
    local features="$2"
    local description="$3"
    local output_file="$RUN_DIR/${package}_cargo_bench.txt"

    print_step "Running cargo bench for $description..."
    print_info "Output: $output_file"

    {
        echo "Cargo Benchmark: $description"
        echo "Package: $package"
        echo "Features: $features"
        echo "Timestamp: $(date)"
        echo ""
        echo "Results:"
        echo "--------"
    } > "$output_file"

    if cargo bench -p "$package" $features 2>&1 | tee -a "$output_file"; then
        print_success "$description cargo benchmarks completed"
        return 0
    else
        print_warn "$description cargo benchmarks failed (may not have benchmarks)"
        return 0  # Don't fail if no benchmarks exist
    fi
}

RESULTS=()

# Sharing/Crypto Benchmarks (no server needed)
if [ "$RUN_SHARING" = true ]; then
    print_header "Sharing & Crypto Benchmarks"

    # Core benchmarks
    run_cargo_benchmarks "shardlm-v2-core" "--features cuda" "Core Operations"

    # Sharing benchmarks
    run_cargo_benchmarks "shardlm-v2-sharing" "--features cuda" "Secret Sharing"

    # OT benchmarks
    run_cargo_benchmarks "shardlm-ot" "" "Oblivious Transfer"

    RESULTS+=("Sharing/Crypto Benchmarks: DONE")
fi

# V2 Benchmarks
if [ "$RUN_V2" = true ]; then
    print_header "V2 Benchmarks"

    if ! check_server; then
        start_server "cuda" "V2"
    fi

    if run_inference_benchmark "v2" "V2 Secure Inference"; then
        RESULTS+=("V2 Inference: DONE")
    else
        RESULTS+=("V2 Inference: FAIL")
    fi
fi

# V3 Benchmarks
if [ "$RUN_V3" = true ]; then
    print_header "V3 Benchmarks (Baseline)"

    if ! check_server; then
        start_server "cuda" "V3"
    fi

    if run_inference_benchmark "v3" "V3 Secure Inference (Baseline)"; then
        RESULTS+=("V3 Inference: DONE")
    else
        RESULTS+=("V3 Inference: FAIL")
    fi
fi

# V3-OT Benchmarks
if [ "$RUN_V3_OT" = true ]; then
    print_header "V3-OT Benchmarks"

    stop_server
    if ! check_server; then
        start_server "cuda" "V3-OT"
    fi

    if run_inference_benchmark "v3-ot" "V3-OT Secure Inference"; then
        RESULTS+=("V3-OT Inference: DONE")
    else
        RESULTS+=("V3-OT Inference: FAIL")
    fi
fi

# V3-MPC Benchmarks
if [ "$RUN_V3_MPC" = true ]; then
    print_header "V3-MPC Benchmarks"

    stop_server
    if ! check_server; then
        start_server "cuda,mpc-secure" "V3-MPC"
    fi

    if run_inference_benchmark "v3-mpc" "V3-MPC Secure Inference"; then
        RESULTS+=("V3-MPC Inference: DONE")
    else
        RESULTS+=("V3-MPC Inference: FAIL")
    fi
fi

# V3-CC Benchmarks
if [ "$RUN_V3_CC" = true ]; then
    print_header "V3-CC Benchmarks"

    stop_server
    if ! check_server; then
        start_server "h100-cc,cuda" "V3-CC"
    fi

    if run_inference_benchmark "v3-cc" "V3-CC Secure Inference"; then
        RESULTS+=("V3-CC Inference: DONE")
    else
        RESULTS+=("V3-CC Inference: FAIL")
    fi
fi

# Generate summary report
print_header "Generating Summary Report"

SUMMARY_FILE="$RUN_DIR/summary.txt"
{
    echo "ShardLM Benchmark Summary"
    echo "========================="
    echo ""
    echo "Run: $TIMESTAMP"
    echo "GPU: $GPU_NAME"
    echo "Model: $MODEL_ARCH"
    echo ""
    echo "Results:"
    for result in "${RESULTS[@]}"; do
        echo "  - $result"
    done
    echo ""
    echo "Output files:"
    ls -la "$RUN_DIR"
} > "$SUMMARY_FILE"

# Create latest symlink
ln -sfn "$TIMESTAMP" "$RESULTS_DIR/latest"

# Summary
print_header "Benchmark Summary"

echo "Results:"
for result in "${RESULTS[@]}"; do
    if [[ "$result" == *"FAIL"* ]]; then
        echo -e "  ${RED}✗${NC} $result"
    else
        echo -e "  ${GREEN}✓${NC} $result"
    fi
done

echo ""
print_info "Results saved to: $RUN_DIR"
print_info "Latest symlink: $RESULTS_DIR/latest"
print_success "Benchmark run complete!"
