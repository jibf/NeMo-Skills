#!/bin/bash

# prepare data
python -m nemo_skills.dataset.prepare nexusbench

# Configuration
BASE_URL="YOUR-BASE-URL"
MODEL="openai/gpt-4o-20240806"
API_KEY="YOUR-API-KEY"
OUTPUT_BASE_DIR="./nexusbench_results"

# Create output directory
mkdir -p ${OUTPUT_BASE_DIR}

# List of all NexusBench tasks
TASKS=(
    "nvd_library"
    "virustotal"
    "it_type0"
    "it_type1"
    "ticket_tracking"
    "tmi_hallucination"
    "climate"
    "cvecpe"
    "virustotal_agentic"
    "langchain_relational"
    "langchain_math"
    "multiverse_math_hard"
    "langchain_typewriter_hard"
    "langchain_multitool_typewriter_hard"
)

for task in "${TASKS[@]}"; do
    echo "[$(date +%H:%M:%S)] Processing: ${task}"

    INPUT_FILE="nemo_skills/dataset/nexusbench/${task}/test.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${task}"
    OUTPUT_FILE="${OUTPUT_DIR}/output.jsonl"

    # Create task output directory
    mkdir -p ${OUTPUT_DIR}

    # Check if input file exists
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "  ✗ ERROR: Input file not found: ${INPUT_FILE}"
        echo "  Run 'python -m nemo_skills.dataset.prepare nexusbench' first"
        continue
    fi

    # Get sample count
    SAMPLE_COUNT=$(wc -l < ${INPUT_FILE})
    echo "  Samples: ${SAMPLE_COUNT}"

    # Run inference (using nemoskills conda environment)
    python -m nemo_skills.inference.eval.nexusbench \
        ++input_file=${INPUT_FILE} \
        ++output_file=${OUTPUT_FILE} \
        ++inference.temperature=0.0 \
        ++server.server_type=openai \
        ++server.base_url=${BASE_URL} \
        ++server.model=${MODEL} \
        "++server.api_key=${API_KEY}" \
        ++eval_type=nexusbench

    if [ $? -eq 0 ]; then
        echo "  ✓ Completed"
        # Show metrics if available
        if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
            echo "  Metrics: ${OUTPUT_DIR}/metrics.json"
            # Extract and display accuracy if available
            if command -v jq &> /dev/null; then
                ACCURACY=$(jq -r '.aggregated.accuracy // empty' ${OUTPUT_DIR}/metrics.json 2>/dev/null)
                if [ ! -z "$ACCURACY" ]; then
                    echo "  Accuracy: ${ACCURACY}"
                fi
            fi
        fi
    else
        echo "  ✗ Failed"
    fi
    echo ""
done


# Aggregate all scores
if [ -d "${OUTPUT_BASE_DIR}" ]; then
    echo "Aggregating scores across all benchmarks..."
    python -m nemo_skills.dataset.nexusbench.nexusbench_score ${OUTPUT_BASE_DIR}
    echo ""
    echo "To view individual results:"
    echo "  cat ${OUTPUT_BASE_DIR}/{task_name}/metrics.json"
fi
