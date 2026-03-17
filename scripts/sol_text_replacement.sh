#!/bin/bash
#SBATCH --job-name=interwhen_text_replace_example
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --time=0-00:30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=sg025
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=tashahee@asu.edu

set -euo pipefail

LOG_DIR="slurm_logs/${SLURM_JOB_ID:-manual_$(date +%s)}"
mkdir -p "$LOG_DIR"
OUT_LOG_FILE="$LOG_DIR/text_replacement_${SLURM_JOB_ID:-na}.out"
ERR_LOG_FILE="$LOG_DIR/text_replacement_${SLURM_JOB_ID:-na}.err"
exec >"$OUT_LOG_FILE" 2>"$ERR_LOG_FILE"

module purge
module load mamba
source activate interwhen

PROJECT_DIR="/scratch/$USER/interwhen"

# Configure model identity and weights path.
# Keep MODEL_NAME and MODEL_PATH aligned to the same checkpoint family.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
MODEL_ROOT="${MODEL_ROOT:-/data/datasets/community/huggingface/models--Qwen--Qwen3-235B-A22B-Thinking-2507}"
MODEL_PATH="${MODEL_PATH:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
VLLM_CONTAINER="${VLLM_CONTAINER:-/home/$USER/vllm-latest.sif}"

if [ -z "$MODEL_PATH" ]; then
    if [ -f "$MODEL_ROOT/refs/main" ]; then
        SNAPSHOT_ID="$(cat "$MODEL_ROOT/refs/main")"
        MODEL_PATH="$MODEL_ROOT/snapshots/$SNAPSHOT_ID"
    elif [ -d "$MODEL_ROOT/snapshots" ]; then
        MODEL_PATH="$(ls -1dt "$MODEL_ROOT"/snapshots/* 2>/dev/null | head -n 1 || true)"
    fi
fi

if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Could not resolve MODEL_PATH. Set MODEL_PATH explicitly." >&2
    exit 1
fi

if [ ! -f "$VLLM_CONTAINER" ]; then
    echo "ERROR: VLLM_CONTAINER not found at $VLLM_CONTAINER" >&2
    exit 1
fi

echo "MODEL_NAME: $MODEL_NAME"
echo "MODEL_PATH: $MODEL_PATH"

PORT=$(shuf -i 10000-65000 -n 1)
while ss -tuln | grep -q ":$PORT "; do
    PORT=$(shuf -i 10000-65000 -n 1)
done
echo "Assigned randomized free port: $PORT"

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $VLLM_PID)"
        kill "$VLLM_PID" || true
    fi
}
trap cleanup EXIT

MAX_START_RETRIES=5
START_ATTEMPT=1

while [ "$START_ATTEMPT" -le "$MAX_START_RETRIES" ]; do
    echo "Starting vLLM (attempt $START_ATTEMPT/$MAX_START_RETRIES)"

    apptainer run --nv --bind /data:/data "$VLLM_CONTAINER" \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --port "$PORT" &

    VLLM_PID=$!
    sleep 5

    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM process is alive (PID: $VLLM_PID)"
        break
    fi

    echo "vLLM failed to start, retrying in 15s"
    sleep 15
    START_ATTEMPT=$((START_ATTEMPT + 1))
done

if [ "$START_ATTEMPT" -gt "$MAX_START_RETRIES" ]; then
    echo "CRITICAL ERROR: vLLM failed to boot after $MAX_START_RETRIES attempts" >&2
    exit 1
fi

echo "Waiting for vLLM readiness on port $PORT"
MAX_URL_RETRIES=60
COUNTER=0
while [ "$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models")" != "200" ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "CRITICAL ERROR: vLLM crashed while loading weights" >&2
        exit 1
    fi

    sleep 10
    COUNTER=$((COUNTER + 1))
    if [ "$COUNTER" -ge "$MAX_URL_RETRIES" ]; then
        echo "ERROR: vLLM did not become ready within timeout" >&2
        exit 1
    fi
done
echo "vLLM is ready"

cd "$PROJECT_DIR"

export INTERWHEN_MODEL_NAME="$MODEL_NAME"
export INTERWHEN_TOKENIZER_NAME="$MODEL_PATH"
export INTERWHEN_PORT="$PORT"
export INTERWHEN_OUTPUT_FILE="$PROJECT_DIR/output_${SLURM_JOB_ID:-manual}.txt"

echo "Running text replacement example"
python ./examples/text_replacement_example.py

echo "Completed successfully"