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

"""DrafterBench evaluation module."""

import json
import logging
import sys
import os
import datetime

# Add dataset path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "drafterbench"))

from tqdm import tqdm
import numpy as np

from nemo_skills.utils import get_logger_name, unroll_files

# Import DrafterBench evaluation modules
try:
    from collect_result import collect_result, process_code, execute_code
    from metric import ground_check, cross_check
    from utils.types import Score_builder
except ImportError as e:
    logging.error(f"Failed to import DrafterBench modules: {e}")
    logging.error("Make sure DrafterBench evaluation files are in nemo_skills/dataset/drafterbench/")
    raise

LOG = logging.getLogger(get_logger_name(__file__))

TASK_SETS = [
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

def eval_drafterbench(cfg):
    """Evaluate DrafterBench results."""

    for file in unroll_files(cfg.input_files):
        LOG.info(f"Evaluating file: {file}")
        with open(file, "rt", encoding="utf-8") as fin:
            data = [json.loads(line) for line in fin]

        with open(file.replace(".json", "_score.json"), "wt", encoding="utf-8") as fout:
            for sample in tqdm(data, desc="Evaluating samples"):
                # Extract data from sample
                test_code = sample.get("generation", "")
                ground_code = sample.get("groundtruth", "")
                precise_mode = "precise" if sample.get("precise_vague") == "Precise" else "vaguely"

                # Process codes
                test_code = process_code(test_code)
                test_info = execute_code(test_code)

                ground_code = process_code(ground_code)
                ground_info = execute_code(ground_code)

                # Calculate ground truth metrics
                ground_details = ground_check(ground_info)

                # Build score
                if test_info:
                    # Cross-check generated code against ground truth
                    score_builder = Score_builder().ground_fill(ground_details)
                    score_builder = score_builder.result(*cross_check(ground_info, test_info, precise_mode))
                else:
                    # Code failed to execute
                    score_builder = Score_builder().ground_fill(ground_details).fail()

                # Collect and format results
                result = collect_result(score_builder)

                # Add evaluation results to sample
                sample.update(result)
                fout.write(json.dumps(sample) + "\n")
        save_dir = os.path.dirname(file)
        generate_score(data, save_dir)

def generate_score(eval_results, save_dir):
    eval_list = list(eval_results)
    task_rewards = {}
    for task in TASK_SETS:
        rewards = [
            x["Task_score"]["Task_score"] for x in eval_list if x["task_type"] == task
        ]
        task_rewards.update({task: np.average(rewards)})
    average_task_rewards = np.average([task_rewards[x] for x in TASK_SETS])
    comprehensive_rewards = (
        average_task_rewards - (100 - min([task_rewards[x] for x in TASK_SETS])) / 12
    )
    structured_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["structured_unstructured"] == "Structured"
    ]
    unstrctured_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["structured_unstructured"] == "Unstructured"
    ]
    precise_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["precise_vague"] == "Precise"
    ]
    vague_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["precise_vague"] == "Vague"
    ]
    complete_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["complete_incomplete"] == "Complete"
    ]
    error_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["complete_incomplete"] == "Error"
    ]
    single_OB_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["single_multiple_objects"] == "Single_Object"
    ]
    multiple_OB_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["single_multiple_objects"] == "Multiple_Objects"
    ]
    single_OP_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["single_multiple_operations"] == "Single_Operation"
    ]
    multiple_OP_rewards = [
        x["Task_score"]["Task_score"]
        for x in eval_list
        if x["single_multiple_operations"] == "Multiple_Operations"
    ]
    average_structured_rewards = (
        np.average(structured_rewards) if structured_rewards else "NaN"
    )
    average_unstrctured_rewards = (
        np.average(unstrctured_rewards) if unstrctured_rewards else "NaN"
    )
    average_precise_rewards = np.average(precise_rewards) if precise_rewards else "NaN"
    average_vague_rewards = np.average(vague_rewards) if vague_rewards else "NaN"
    average_complete_rewards = (
        np.average(complete_rewards) if complete_rewards else "NaN"
    )
    average_error_rewards = np.average(error_rewards) if error_rewards else "NaN"
    average_single_OB_rewards = (
        np.average(single_OB_rewards) if single_OB_rewards else "NaN"
    )
    average_multiple_OB_rewards = (
        np.average(multiple_OB_rewards) if multiple_OB_rewards else "NaN"
    )
    average_single_OP_rewards = (
        np.average(single_OP_rewards) if single_OP_rewards else "NaN"
    )
    average_multiple_OP_rewards = (
        np.average(multiple_OP_rewards) if multiple_OP_rewards else "NaN"
    )

    print("Average reward in different metrics:\n")
    reward = (
        f"Structured language: {average_structured_rewards}\n"
        f"Unstructured language: {average_unstrctured_rewards}\n"
        f"Precise detail: {average_precise_rewards}\n"
        f"Vague detail: {average_vague_rewards}\n"
        f"Complete instruction: {average_complete_rewards}\n"
        f"Incomplete (error) instruction: {average_error_rewards}\n"
        f"Single object: {average_single_OB_rewards}\n"
        f"Multiple objects: {average_multiple_OB_rewards}\n"
        f"Single operation: {average_single_OP_rewards}\n"
        f"Multiple operations: {average_multiple_OP_rewards}\n"
        f"Average tasks: {average_task_rewards}\n"
        f"Comprehensive rewards: {comprehensive_rewards}"
    )
    
    print(reward)
    text_result_path = f"{save_dir}/drafterbench_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"
    with open(text_result_path, "w", encoding="utf-8") as w:
        w.write(reward)
    print(f"The experiment result has been saved in {text_result_path}.")