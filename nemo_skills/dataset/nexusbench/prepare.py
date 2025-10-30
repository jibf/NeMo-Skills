# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

HF_DATASET_MAPPING = {
    "nvd_library": "Nexusflow/NVDLibraryBenchmark",
    "virustotal": "Nexusflow/VirusTotalBenchmark",
    "it_type0": "Nexusflow/ITType0Benchmark",
    "it_type1": "Nexusflow/ITType1Benchmark",
    "ticket_tracking": "Nexusflow/TicketTrackingBenchmark",
    "tmi_hallucination": "Nexusflow/HallucinationTMIBenchmark",
    "climate": "Nexusflow/ClimateAPIBenchmark",
    "cvecpe": "Nexusflow/CVECPEAPIBenchmark",
    "virustotal_agentic": "Nexusflow/VirusTotalAgentic",
    "langchain_relational": "Nexusflow/LangChainRelational",
    "langchain_math": "Nexusflow/LangChainMathBenchmark",
    "multiverse_math_hard": "Nexusflow/MultiverseMathHard",
    "langchain_typewriter_hard": "Nexusflow/Langchain-Typewriter-hard-no-whitespaces",
    "langchain_multitool_typewriter_hard": "Nexusflow/LangChainMultitoolTypeWriterHard",
}

AGENT_BENCHMARKS = {
    "langchain_math",
    "climate",
    "cvecpe",
    "virustotal_agentic",
    "langchain_relational",
    "multiverse_math_hard",
    "langchain_typewriter_hard",
    "langchain_multitool_typewriter_hard",
}

# Field mapping for each benchmark based on their HuggingFace dataset structure
FIELD_MAPPING = {
    "nvd_library": {"question_field": "Input", "ground_truth_field": "Output"},
    "virustotal": {"question_field": "Input", "ground_truth_field": "Output"},
    "it_type0": {"question_field": "Input", "ground_truth_field": "Output"},
    "it_type1": {"question_field": "Input", "ground_truth_field": "Output"},
    "ticket_tracking": {"question_field": "Input", "ground_truth_field": "Output"},
    "climate": {"question_field": "Input", "ground_truth_field": "Output"},
    "cvecpe": {"question_field": "Input", "ground_truth_field": "Output"},
    "langchain_math": {"question_field": "inputs.question", "ground_truth_field": "outputs.reference"},
    "tmi_hallucination": {"question_field": "user_query", "ground_truth_field": "original_ground_truth"},
    "virustotal_agentic": {"question_field": "generated_question", "ground_truth_field": "fncall"},
    "langchain_relational": {"question_field": "user_query", "ground_truth_field": "ground_truth"},
    "multiverse_math_hard": {"question_field": "prompt", "ground_truth_field": "ground_truth"},
    "langchain_typewriter_hard": {"question_field": "answer", "ground_truth_field": "answer"},
    "langchain_multitool_typewriter_hard": {"question_field": "answer", "ground_truth_field": "answer"},
}

SPLIT_CONFIG_TEMPLATE = """DATASET_GROUP = "tool"
METRICS_TYPE = "nexusbench"
EVAL_ARGS = "++eval_type=nexusbench"
GENERATION_ARGS = ""
GENERATION_MODULE = "nemo_skills.inference.eval.nexusbench"
"""


def get_field_value(sample, field_path):
    """Extract value from sample using dot-notation field path."""
    if "." in field_path:
        parts = field_path.split(".")
        value = sample
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    else:
        return sample.get(field_path)


def prepare_split(split_name, output_dir):
    """Prepare a single benchmark split."""
    if split_name not in HF_DATASET_MAPPING:
        LOG.warning(f"No HuggingFace mapping for {split_name}, skipping")
        return

    LOG.info(f"Preparing {split_name}...")
    try:
        dataset = load_dataset(HF_DATASET_MAPPING[split_name], split="train")
        # Typewriter benchmarks only use first 30 samples
        if split_name in ["langchain_typewriter_hard", "langchain_multitool_typewriter_hard"]:
            dataset = dataset.select(range(30))
    except Exception as e:
        LOG.error(f"Failed to load dataset for {split_name}: {e}")
        return

    split_dir = output_dir / split_name
    split_dir.mkdir(exist_ok=True)

    with open(split_dir / "__init__.py", "w") as f:
        f.write(SPLIT_CONFIG_TEMPLATE)

    is_agent = split_name in AGENT_BENCHMARKS
    max_turns = 20 if is_agent else 1

    # Get field mapping for this benchmark
    field_mapping = FIELD_MAPPING.get(split_name, {})
    question_field = field_mapping.get("question_field")
    ground_truth_field = field_mapping.get("ground_truth_field")

    if not question_field or not ground_truth_field:
        LOG.error(f"Missing field mapping for {split_name}")
        return

    successful_samples = 0
    with open(split_dir / "test.jsonl", "w") as f:
        for idx, sample in enumerate(dataset):
            entry = {
                "id": f"{split_name}_{idx}",
                "single_turn": not is_agent,
                "max_turns": max_turns,
                "benchmark_type": split_name,
            }

            # Extract question
            question = get_field_value(sample, question_field)
            if question is None:
                LOG.debug(f"Skipping sample {idx} - no question field '{question_field}'")
                continue
            entry["question"] = str(question)

            # Extract ground truth
            ground_truth = get_field_value(sample, ground_truth_field)
            if ground_truth is None:
                LOG.debug(f"Skipping sample {idx} - no ground_truth field '{ground_truth_field}'")
                continue

            # Handle special cases
            if split_name == "tmi_hallucination":
                # Extract args_to_avoid from original/modified calls
                original_call = sample.get("original_ground_truth", "")
                correct_call = sample.get("modified_correct_ground_truth", "")
                try:
                    import ast

                    original_ast = ast.parse(original_call).body[0].value
                    correct_ast = ast.parse(correct_call).body[0].value
                    original_args = {kw.arg for kw in original_ast.keywords}
                    correct_args = {kw.arg for kw in correct_ast.keywords}
                    args_to_avoid = list(original_args - correct_args)
                    entry["ground_truth"] = args_to_avoid
                    entry["json_tools"] = sample.get("json_tools", "")
                except Exception as e:
                    LOG.debug(f"Failed to parse TMI sample {idx}: {e}")
                    continue
            elif split_name == "nvd_library":
                # Strip the "r = nvdlib." prefix
                entry["ground_truth"] = str(ground_truth).replace("r = nvdlib.", "")
            elif split_name == "multiverse_math_hard":
                # Execute ground_truth to get numerical results
                if isinstance(ground_truth, str):
                    try:
                        from nemo_skills.dataset.nexusbench.tools.langchain_math import (
                            multiply,
                            divide,
                            add,
                            sin,
                            cos,
                            subtract,
                            power,
                            log,
                            pi,
                            negate,
                            return_constant,
                        )

                        gt_results = []
                        for g in ground_truth.split(";"):
                            g = g.strip()
                            if g:
                                # pylint: disable=eval-used
                                result = eval(
                                    g,
                                    {
                                        "multiply": multiply,
                                        "divide": divide,
                                        "add": add,
                                        "sin": sin,
                                        "cos": cos,
                                        "subtract": subtract,
                                        "power": power,
                                        "log": log,
                                        "pi": pi,
                                        "negate": negate,
                                        "return_constant": return_constant,
                                    },
                                )
                                gt_results.append(result)
                        entry["ground_truth"] = gt_results
                    except Exception as e:
                        LOG.debug(f"Failed to evaluate multiverse_math ground_truth for sample {idx}: {e}")
                        continue
                else:
                    entry["ground_truth"] = ground_truth
            else:
                entry["ground_truth"] = ground_truth

            f.write(json.dumps(entry) + "\n")
            successful_samples += 1

    LOG.info(f"  Prepared {split_name}: {successful_samples}/{len(dataset)} samples")


def main(args):
    setup_logging()
    output_dir = Path(__file__).absolute().parent

    if args.splits:
        splits_to_prepare = args.splits.split(",")
    else:
        splits_to_prepare = list(HF_DATASET_MAPPING.keys())

    LOG.info(f"Preparing {len(splits_to_prepare)} splits")

    for split_name in splits_to_prepare:
        try:
            prepare_split(split_name, output_dir)
        except Exception as e:
            LOG.error(f"Failed to prepare {split_name}: {e}", exc_info=True)

    LOG.info("NexusBench data preparation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NexusBench datasets")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated list of splits to prepare (default: all)",
    )
    args = parser.parse_args()
    main(args)
