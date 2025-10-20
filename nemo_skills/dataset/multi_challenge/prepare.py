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

import json
import logging
from collections import defaultdict
from pathlib import Path

LOG = logging.getLogger(__file__)

# Mapping from AXIS names to split directory names
AXIS_TO_SPLIT = {
    "INFERENCE_MEMORY": "inference_memory",
    "INSTRUCTION_RETENTION": "instruction_retention",
    "RELIABLE_VERSION_EDITING": "reliable_version_editing",
    "SELF_COHERENCE": "self_coherence",
}


def prepare_multi_challenge_data(output_path=None):
    """Prepare MultiChallenge dataset by splitting data into task type subdirectories."""
    script_dir = Path(__file__).parent

    if output_path is None:
        output_path = script_dir
    else:
        output_path = Path(output_path)

    # Input file is in the same directory as this script
    input_file = script_dir / "benchmark_questions.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"The MultiChallenge benchmark_questions.jsonl file should be included in the repository."
        )

    axis_data = defaultdict(list)

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line)
            converted_data = {
                "question_id": data["QUESTION_ID"],
                "axis": data["AXIS"],
                "conversation": data["CONVERSATION"],
                "target_question": data["TARGET_QUESTION"],
                "pass_criteria": data["PASS_CRITERIA"],
                "expected_answer": None,
            }
            axis_data[data["AXIS"]].append(converted_data)

    for axis, axis_name in AXIS_TO_SPLIT.items():
        if axis not in axis_data:
            LOG.warning(f"No data found for axis: {axis}")
            continue

        split_dir = output_path / axis_name
        split_dir.mkdir(parents=True, exist_ok=True)
        output_file = split_dir / "test.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for item in axis_data[axis]:
                f_out.write(json.dumps(item) + '\n')

        print(f"  {axis_name:30s}: {len(axis_data[axis]):3d} questions → {output_file}")

    print(f"\nMultiChallenge data prepared successfully!")
    print(f"Total: {sum(len(items) for items in axis_data.values())} questions across {len(axis_data)} axes")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        # Default: prepare in the current dataset directory
        output_path = None

    prepare_multi_challenge_data(output_path)
