#!/bin/bash
# ShardLM Test Runner Script
# Runs all tests across the workspace
#
# Usage:
#   ./scripts/run_tests.sh              # Run all tests
#   ./scripts/run_tests.sh --unit       # Run unit tests only
#   ./scripts/run_tests.sh --integration # Run integration tests only
#   ./scripts/run_tests.sh --security   # Run security/adversarial tests only
#   ./scripts/run_tests.sh --package <name> # Run tests for specific package

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

# Parse command line arguments
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_SECURITY=false
RUN_ALL=true
SPECIFIC_PACKAGE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT=true
            RUN_ALL=false
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            RUN_ALL=false
            shift
            ;;
        --security)
            RUN_SECURITY=true
            RUN_ALL=false
            shift
            ;;
        --package|-p)
            SPECIFIC_PACKAGE="$2"
            RUN_ALL=false
            shift 2
            ;;
        --features)
            EXTRA_ARGS="$EXTRA_ARGS --features $2"
            shift 2
            ;;
        --release)
            EXTRA_ARGS="$EXTRA_ARGS --release"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit          Run unit tests only"
            echo "  --integration   Run integration tests only"
            echo "  --security      Run security/adversarial tests only"
            echo "  --package <pkg> Run tests for specific package"
            echo "  --features <f>  Additional cargo features"
            echo "  --release       Run tests in release mode"
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
    RUN_UNIT=true
    RUN_INTEGRATION=true
    RUN_SECURITY=true
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_header "ShardLM Test Runner"

# Check if CUDA is available
HAS_CUDA=false
if command -v nvidia-smi &> /dev/null; then
    HAS_CUDA=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_info "GPU detected: $GPU_NAME"
    CUDA_FEATURES="--features cuda"
else
    print_warn "No GPU detected, running CPU-only tests"
    CUDA_FEATURES=""
fi

RESULTS=()
FAILED=0

# Run tests for a specific package
run_package_tests() {
    local package="$1"
    local features="$2"
    local description="$3"

    print_step "Testing $description..."

    if cargo test -p "$package" $features $EXTRA_ARGS 2>&1; then
        print_success "$description passed"
        RESULTS+=("$description: PASS")
        return 0
    else
        print_error "$description failed"
        RESULTS+=("$description: FAIL")
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Run tests with specific test filter
run_filtered_tests() {
    local filter="$1"
    local features="$2"
    local description="$3"

    print_step "Testing $description..."

    if cargo test $filter $features $EXTRA_ARGS 2>&1; then
        print_success "$description passed"
        RESULTS+=("$description: PASS")
        return 0
    else
        print_error "$description failed"
        RESULTS+=("$description: FAIL")
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Specific package requested
if [ -n "$SPECIFIC_PACKAGE" ]; then
    print_header "Testing Package: $SPECIFIC_PACKAGE"
    run_package_tests "$SPECIFIC_PACKAGE" "$CUDA_FEATURES" "$SPECIFIC_PACKAGE"
else
    # Unit Tests
    if [ "$RUN_UNIT" = true ]; then
        print_header "Unit Tests"

        # Core crate
        run_package_tests "shardlm-v2-core" "$CUDA_FEATURES" "Core (shardlm-v2-core)"

        # Sharing crate
        run_package_tests "shardlm-v2-sharing" "$CUDA_FEATURES" "Sharing (shardlm-v2-sharing)"

        # OT crate
        run_package_tests "shardlm-ot" "" "Oblivious Transfer (shardlm-ot)"

        # Model crate (if CUDA available)
        if [ "$HAS_CUDA" = true ]; then
            run_package_tests "shardlm-v2-model" "--features cuda" "Model (shardlm-v2-model)"
        else
            print_warn "Skipping model tests (requires CUDA)"
        fi

        # CC crate (if H100 available)
        if [ "$HAS_CUDA" = true ]; then
            # Check if H100
            if echo "$GPU_NAME" | grep -qi "H100"; then
                run_package_tests "shardlm-v2-cc" "--features nvidia-cc" "Confidential Computing (shardlm-v2-cc)"
            else
                # Run with software-cc on non-H100 GPUs
                run_package_tests "shardlm-v2-cc" "--features software-cc" "Confidential Computing (shardlm-v2-cc)"
            fi
        fi
    fi

    # Integration Tests
    if [ "$RUN_INTEGRATION" = true ]; then
        print_header "Integration Tests"

        # Server integration tests
        if [ "$HAS_CUDA" = true ]; then
            run_package_tests "shardlm-v2-server" "--features cuda" "Server Integration (shardlm-v2-server)"
        else
            print_warn "Skipping server integration tests (requires CUDA)"
        fi

        # Client tests
        run_package_tests "shardlm-v2-client" "" "Client (shardlm-v2-client)"
    fi

    # Security/Adversarial Tests
    if [ "$RUN_SECURITY" = true ]; then
        print_header "Security & Adversarial Tests"

        # Run adversarial tests specifically
        print_step "Running adversarial tests..."
        if cargo test adversarial $CUDA_FEATURES $EXTRA_ARGS 2>&1; then
            print_success "Adversarial tests passed"
            RESULTS+=("Adversarial Tests: PASS")
        else
            print_error "Adversarial tests failed"
            RESULTS+=("Adversarial Tests: FAIL")
            FAILED=$((FAILED + 1))
        fi

        # Run security-related tests
        print_step "Running security tests..."
        if cargo test security $CUDA_FEATURES $EXTRA_ARGS 2>&1; then
            print_success "Security tests passed"
            RESULTS+=("Security Tests: PASS")
        else
            print_error "Security tests failed"
            RESULTS+=("Security Tests: FAIL")
            FAILED=$((FAILED + 1))
        fi

        # Run MPC tests if feature available
        if [ "$HAS_CUDA" = true ]; then
            print_step "Running MPC protocol tests..."
            if cargo test -p shardlm-v2-sharing mpc --features cuda $EXTRA_ARGS 2>&1; then
                print_success "MPC protocol tests passed"
                RESULTS+=("MPC Protocol Tests: PASS")
            else
                print_error "MPC protocol tests failed"
                RESULTS+=("MPC Protocol Tests: FAIL")
                FAILED=$((FAILED + 1))
            fi
        fi
    fi
fi

# Summary
print_header "Test Summary"

echo "Results:"
for result in "${RESULTS[@]}"; do
    if [[ "$result" == *"PASS"* ]]; then
        echo -e "  ${GREEN}✓${NC} $result"
    else
        echo -e "  ${RED}✗${NC} $result"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    print_success "All tests passed!"
    exit 0
else
    print_error "$FAILED test suite(s) failed"
    exit 1
fi
