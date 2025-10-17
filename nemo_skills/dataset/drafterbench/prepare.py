# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Script to prepare DrafterBench dataset for evaluation."""

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def prepare_drafterbench(output_dir: Path):
    """Download and prepare DrafterBench dataset from HuggingFace.

    Args:
        output_dir: Directory to save the prepared dataset (should be nemo_skills/dataset/drafterbench)
    """
    output_dir = Path(output_dir)

    LOG.info("Downloading DrafterBench dataset from HuggingFace...")
    dataset = load_dataset("Eason666/DrafterBench", "drafter_tasks")

    # Task sets in DrafterBench
    task_sets = [
        "add_table",
        "revise_table",
        "map_table",
        "refresh_table",
        "add_text",
        "revise_text",
        "map_text",
        "refresh_text",
        "add_vector",
        "delete_vector",
        "map_vector",
        "refresh_vector",
    ]

    # Load system prompts
    prompt_dir = output_dir / "prompts"
    system_prompts = {}
    prompt_files = [
        ("1", "add_table.txt"),
        ("2", "revise_table.txt"),
        ("3", "map_table.txt"),
        ("4", "refresh_table.txt"),
        ("5", "add_text.txt"),
        ("6", "revise_text.txt"),
        ("7", "map_text.txt"),
        ("8", "refresh_text.txt"),
        ("9", "add_vector.txt"),
        ("10", "delete_vector.txt"),
        ("11", "map_vector.txt"),
        ("12", "refresh_vector.txt"),
    ]

    LOG.info("Loading system prompts...")
    for task_id, filename in prompt_files:
        prompt_file = prompt_dir / filename
        if prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompts[task_id] = f.read()
        else:
            LOG.warning(f"Prompt file not found: {prompt_file}")

    all_samples = []
    for task_set in task_sets:
        if task_set not in dataset:
            LOG.warning(f"Task set '{task_set}' not found in dataset, skipping...")
            continue

        LOG.info(f"Processing task set: {task_set} ({len(dataset[task_set])} samples)")
        for idx, item in enumerate(dataset[task_set]):
            system_prompt_index = str(task_sets.index(task_set) + 1)
            sample = {
                "id": f"{task_set}_{idx}",
                "task_type": task_set,
                "task_id": item.get("Id", idx),
                "instruction": item["Instruction"],
                "groundtruth": item["Groundtruth"],
                "system_prompt": system_prompts.get(system_prompt_index, ""),
                # Include metadata for evaluation
                "precise_vague": item.get("Precise|Vague", ""),
                "complete_incomplete": item.get("Complete|Incomplete", ""),
                "single_multiple_objects": item.get("Single|Multiple_objects", ""),
                "single_multiple_operations": item.get("Single|Multiple_operations", ""),
                "structured_unstructured": item.get("Structured/Unstructured", ""),
            }
            all_samples.append(sample)

    # Save as JSONL
    output_file = output_dir / "test.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    LOG.info(f"Saved {len(all_samples)} samples to {output_file}")
    LOG.info("DrafterBench dataset preparation complete!")


def main(args):
    # The output_dir should be the drafterbench directory itself
    output_dir = Path(__file__).parent
    prepare_drafterbench(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DrafterBench dataset")
    parser.add_argument("--model_type", type=str, default=None, required=False, help="Not used, for compatibility")
    args = parser.parse_args()

    main(args)
