#!/bin/bash
# MultiChallenge benchmark run script
# Updated to follow BFCL golden standard: generation + evaluation in one step

# Configuration - modify these as needed
BASE_URL="${BASE_URL:-http://5.78.122.79:12500/v1}"
MODEL="${MODEL:-openai/gpt-4o-20240806}"
API_KEY="${API_KEY:-sk-sgl-MH7bEVVJlBp3RT_P5cPQ6-KfC1qJElBRCfTDHy40Ue4}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"  # -1 for full dataset
ATTEMPTS="${ATTEMPTS:-1}"
MAX_WORKERS="${MAX_WORKERS:-16}"  # Parallel workers for LLM judge

# Output directory
OUTPUT_DIR="./multi_challenge_results"

# Task types for MultiChallenge
TASK_TYPES=("inference_memory" "instruction_retention" "reliable_version_editing" "self_coherence")

# Step 1: Generation + Evaluation (integrated)
echo "=========================================="
echo "Step 1: Generation + Evaluation (per task type)"
echo "=========================================="

for task_type in "${TASK_TYPES[@]}"; do
    echo ""
    echo ">>> Processing task type: ${task_type}..."

    python -m nemo_skills.inference.eval.multi_challenge \
        ++input_file=nemo_skills/dataset/multi_challenge/${task_type}/test.jsonl \
        ++output_file=${OUTPUT_DIR}/${task_type}/output.jsonl \
        ++eval_type=multi_challenge \
        ++eval_config.judge_model=${MODEL} \
        ++eval_config.judge_base_url=${BASE_URL} \
        ++eval_config.judge_api_key=${API_KEY} \
        ++eval_config.max_workers=${MAX_WORKERS} \
        ++inference.tokens_to_generate=8192 \
        ++inference.temperature=0.0 \
        ++server.server_type=openai \
        ++server.base_url=${BASE_URL} \
        ++server.model=${MODEL} \
        ++server.api_key=${API_KEY} \
        ++attempts=${ATTEMPTS} \
        ++max_samples=${MAX_SAMPLES}

    if [ $? -ne 0 ]; then
        echo "Error: Generation/Evaluation failed for ${task_type}"
        exit 1
    fi

    echo "✓ Completed: ${task_type}"
done

echo ""
echo "=========================================="
echo "Step 2: Computing overall group score"
echo "=========================================="

# Step 2: Aggregate scores across all task types
python -m nemo_skills.evaluation.compute_group_score \
    ${OUTPUT_DIR}/inference_memory/metrics.json \
    ${OUTPUT_DIR}/instruction_retention/metrics.json \
    ${OUTPUT_DIR}/reliable_version_editing/metrics.json \
    ${OUTPUT_DIR}/self_coherence/metrics.json \
    --score_module nemo_skills.dataset.multi_challenge.multi_challenge_score \
    --save_metrics_file ${OUTPUT_DIR}/overall_metrics.json

