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

"""NexusBench evaluation implementation."""

import json
import logging
from dataclasses import dataclass
from inspect import getmembers, isfunction, signature
from pathlib import Path
from typing import Any, Callable, Dict, List

from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class NexusBenchEvaluatorConfig:
    pass


@dataclass
class FunctionCall:
    name: str
    args: Dict[str, Any]

    def __eq__(self, other: "FunctionCall | Any") -> bool:
        if not isinstance(other, FunctionCall):
            return False
        if self.name != other.name:
            return False
        if self.args.keys() != other.args.keys():
            return False
        for k, v in self.args.items():
            if k not in other.args:
                return False
            if type(v) != type(other.args[k]):
                return False
            if repr(v) != repr(other.args[k]):
                return False
        return True


def run_function_calls(function_calls_str: str, tools: List[Callable]) -> List[FunctionCall] | None:
    """Parse function calls from string and convert to FunctionCall objects for comparison."""
    function_call_list: List[FunctionCall] = []
    locals_to_pass = {
        "function_call_list": function_call_list,
        "FunctionCall": FunctionCall,
        "Dict": Dict,
        "List": List,
        "Any": Any,
    }

    for tool in tools:
        name = tool.__name__
        function_definition = f"""def {name}{signature(tool)}:
    function_call = FunctionCall(name="{name}", args=locals())
    function_call_list.append(function_call)
    return function_call
"""
        # pylint: disable=exec-used
        exec(function_definition, locals_to_pass)

    calls = [c.strip() for c in function_calls_str.split(";") if c.strip()]
    for call in calls:
        try:
            # pylint: disable=eval-used
            eval(call, locals_to_pass)
        except:
            return None

    return function_call_list


def load_tools_for_benchmark(benchmark_type: str) -> List[Callable]:
    """Load tools for a specific benchmark type."""
    try:
        if benchmark_type == "nvd_library":
            from nemo_skills.dataset.nexusbench.tools import nvdlib

            return [nvdlib.searchCVE, nvdlib.searchCPE]
        elif benchmark_type == "virustotal":
            from nemo_skills.dataset.nexusbench.tools import virustotal

            return [func for name, func in getmembers(virustotal, isfunction) if name.startswith("vt_")]
        elif benchmark_type == "it_type0":
            from nemo_skills.dataset.nexusbench.tools import it_hard_0

            return [it_hard_0.match_values]
        elif benchmark_type == "it_type1":
            from nemo_skills.dataset.nexusbench.tools import it_hard_1

            return [it_hard_1.match_values]
        elif benchmark_type == "ticket_tracking":
            from nemo_skills.dataset.nexusbench.tools import ticket_tracking

            return [func for name, func in getmembers(ticket_tracking, isfunction)]
        elif benchmark_type == "tmi_hallucination":
            return []
        elif benchmark_type == "climate":
            from nemo_skills.dataset.nexusbench.tools import climate

            return [func for name, func in getmembers(climate, isfunction)]
        elif benchmark_type == "cvecpe":
            from nemo_skills.dataset.nexusbench.tools import cvecpe

            return [func for name, func in getmembers(cvecpe, isfunction)]
        elif benchmark_type == "virustotal_agentic":
            from nemo_skills.dataset.nexusbench.tools import virustotal_nested

            return [func for name, func in getmembers(virustotal_nested, isfunction)]
        elif benchmark_type.startswith("langchain_"):
            if "relational" in benchmark_type:
                from nemo_skills.dataset.nexusbench.tools import relational

                return [func for name, func in getmembers(relational, isfunction)]
            elif "math" in benchmark_type:
                from nemo_skills.dataset.nexusbench.tools import langchain_math

                return [func for name, func in getmembers(langchain_math, isfunction)]
            elif "multitool_typewriter" in benchmark_type:
                from nemo_skills.dataset.nexusbench.tools import multitool_typewriter_hard

                return [func for name, func in getmembers(multitool_typewriter_hard, isfunction)]
            elif "typewriter" in benchmark_type:
                from nemo_skills.dataset.nexusbench.tools import typewriter_hard

                return [func for name, func in getmembers(typewriter_hard, isfunction)]
        elif benchmark_type == "multiverse_math_hard":
            from nemo_skills.dataset.nexusbench.tools import langchain_math

            return [func for name, func in getmembers(langchain_math, isfunction)]

        LOG.warning(f"Unknown benchmark type: {benchmark_type}")
        return []
    except Exception as e:
        LOG.error(f"Failed to load tools for {benchmark_type}: {e}")
        return []


def check_single_turn_correctness(sample, tools):
    """Check correctness for single-turn benchmarks by comparing function calls."""
    generation = sample.get("generation", "")
    ground_truth = sample.get("ground_truth", "")

    if not generation or not ground_truth:
        return False

    # Flatten generation if it's a list
    if isinstance(generation, list):
        generation_str = []
        for turn in generation:
            if isinstance(turn, list):
                generation_str.extend(turn)
            else:
                generation_str.append(turn)
        generation = ";".join(g for g in generation_str if g)

    if not generation:
        return False

    # Run model's generation
    model_result = run_function_calls(generation, tools)
    if not model_result or len(model_result) == 0:
        return False
    model_call = model_result[0]

    # Run ground truth
    gt_result = run_function_calls(ground_truth, tools)
    if not gt_result or len(gt_result) == 0:
        return False
    gt_call = gt_result[0]

    # Compare using FunctionCall.__eq__
    return gt_call == model_call


def check_climate_correctness(sample, tools):
    """Check correctness for Climate benchmark."""
    execution_results = sample.get("execution_results", [])
    generation = sample.get("generation", [])
    ground_truth = sample.get("ground_truth", "")

    if not generation or not ground_truth:
        return False

    # Flatten generation and check for END_TOKEN_PREDICTED
    model_calls_list = []
    for turn in generation:
        if isinstance(turn, list):
            model_calls_list.extend(turn)
        else:
            model_calls_list.append(turn)

    # Must end with END_TOKEN_PREDICTED
    if not model_calls_list or model_calls_list[-1] != "END_TOKEN_PREDICTED":
        return False

    if not execution_results:
        return False

    # Execute ground truth to get expected results
    ground_truth_list = [func.strip() for func in ground_truth.split(";") if func.strip()]
    ground_truth_calls = []
    try:
        from nemo_skills.dataset.nexusbench.tools.climate import (
            get_latitude_longitude,
            get_current_location,
            find_nearby_stations,
            get_nearest_station_id,
            get_timezone,
            get_hourly_observation,
            subtract_time_delta,
            get_current_time_at_location,
        )

        for call in ground_truth_list:
            # pylint: disable=eval-used
            result = eval(
                call,
                {
                    "get_latitude_longitude": get_latitude_longitude,
                    "get_current_location": get_current_location,
                    "find_nearby_stations": find_nearby_stations,
                    "get_nearest_station_id": get_nearest_station_id,
                    "get_timezone": get_timezone,
                    "get_hourly_observation": get_hourly_observation,
                    "subtract_time_delta": subtract_time_delta,
                    "get_current_time_at_location": get_current_time_at_location,
                },
            )
            ground_truth_calls.append(result)
    except Exception as e:
        LOG.debug(f"Error evaluating climate ground truth: {e}")
        return False

    # Flatten execution results
    step_calls = []
    for result in execution_results:
        if isinstance(result, list):
            step_calls.extend(result)
        else:
            step_calls.append(result)

    if not step_calls:
        return False

    # Check if all ground truth results are in execution results
    return all(element in step_calls for element in ground_truth_calls)


def check_cvecpe_correctness(sample, tools):
    """Check correctness for CVECPE by comparing execution output with ground truth."""
    execution_results = sample.get("execution_results", [])
    ground_truth = sample.get("ground_truth", "")

    if not execution_results or not ground_truth:
        return False

    # Get the final output
    output = None
    if isinstance(execution_results, list):
        if not execution_results:
            return False
        for turn in reversed(execution_results):
            if isinstance(turn, list):
                for result in reversed(turn):
                    if result:
                        output = result
                        break
            elif turn:
                output = turn
                break
            if output is not None:
                break
    else:
        output = execution_results

    if output is None:
        return False

    # Execute ground_truth to get expected output
    try:
        from nemo_skills.dataset.nexusbench.tools.cvecpe import (
            searchCPE,
            searchCVE,
            summarize_cvecpes,
            compare_cvecpes,
            verify_and_process_data_range_start,
            verify_and_process_data_range_end,
            search_backup_keywords,
            count_cvecpe_items,
            mergeCVEs,
            mergeCPEs,
            getCPEName,
            get_first_object_from_list,
            countCVEsBySeverity,
            sortCVEsByCVSSv3Score,
            sortCVEsByCVSSv2Score,
            sortCVEsByModDate,
            sortCPEsByLastMod,
            filterDeprecatedCPEs,
            filterCVEsBySeverity,
            filterCVEByLanguage,
        )

        expected_output = eval(ground_truth)
        return expected_output == output
    except Exception as e:
        LOG.debug(f"Error evaluating cvecpe ground truth: {e}")
        return False


def check_virustotal_agentic_correctness(sample, tools):
    """Check correctness for VirusTotalAgentic."""
    execution_results = sample.get("execution_results", [])
    generation = sample.get("generation", [])
    ground_truth = sample.get("ground_truth", "")

    if not generation:
        return False

    # Handle list ground_truth
    if isinstance(ground_truth, list):
        if not ground_truth:
            return False
        ground_truth_str = ground_truth[0] if isinstance(ground_truth[0], str) else str(ground_truth[0])
    else:
        ground_truth_str = ground_truth

    if not ground_truth_str:
        return False

    # Flatten generation and check for END_TOKEN_PREDICTED
    model_calls_list = []
    for turn in generation:
        if isinstance(turn, list):
            model_calls_list.extend(turn)
        else:
            model_calls_list.append(turn)

    # Must end with END_TOKEN_PREDICTED
    if not model_calls_list or model_calls_list[-1] != "END_TOKEN_PREDICTED":
        return False

    if not execution_results:
        return False

    # Execute ground truth to get expected results
    ground_truth_calls = []
    try:
        from nemo_skills.dataset.nexusbench.tools import virustotal_nested

        vt_functions = {name: func for name, func in getmembers(virustotal_nested, isfunction)}

        # pylint: disable=eval-used
        result = eval(ground_truth_str, vt_functions)
        if isinstance(result, list):
            ground_truth_calls.extend(result)
        else:
            ground_truth_calls.append(result)
    except Exception as e:
        LOG.debug(f"Error evaluating virustotal_agentic ground truth: {e}")
        return False

    # Flatten execution results
    step_calls = []
    for result in execution_results:
        if isinstance(result, list):
            step_calls.extend(result)
        else:
            step_calls.append(result)

    if not step_calls:
        return False

    # Check if all ground truth results are in execution results
    return all(element in step_calls for element in ground_truth_calls)


def check_langchain_math_correctness(sample):
    """Check correctness for LangChainMath."""
    execution_results = sample.get("execution_results", [])
    ground_truth = sample.get("ground_truth")

    if not execution_results:
        return False

    # Get final result
    if isinstance(execution_results, list):
        if not execution_results:
            return False
        if isinstance(execution_results[-1], list):
            if not execution_results[-1]:
                return False
            final_result = execution_results[-1][-1]
        else:
            final_result = execution_results[-1]
    else:
        final_result = execution_results

    if isinstance(final_result, str) and "Error" in final_result:
        return False

    # Compare with ground truth
    try:
        if isinstance(ground_truth, (int, float)):
            return abs(float(final_result) - float(ground_truth)) < 1e-6
        else:
            return str(final_result).strip() == str(ground_truth).strip()
    except:
        return False


def check_langchain_relational_correctness(sample, tools):
    """Check correctness for LangChainRelational."""
    execution_results = sample.get("execution_results", [])
    generation = sample.get("generation", [])
    ground_truth = sample.get("ground_truth", [])

    if not generation:
        return False

    # Flatten generation
    model_calls_list = []
    for turn in generation:
        if isinstance(turn, list):
            model_calls_list.extend(turn)
        else:
            model_calls_list.append(turn)

    if not model_calls_list:
        return False

    has_end_token = model_calls_list[-1] == "END_TOKEN_PREDICTED"

    # If no END token and no execution results, it's incorrect
    if not has_end_token and not execution_results:
        return False

    # Convert execution results to strings for comparison
    step_calls = []
    for result in execution_results:
        if isinstance(result, list):
            for r in result:
                step_calls.append(str(r))
        else:
            step_calls.append(str(result))

    # Handle list ground_truth
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]

    # Check if ground_truth values are in execution results
    if step_calls:
        if all(str(element) in step_calls or element in step_calls for element in ground_truth):
            return True
        if str(ground_truth) in [str(call) for call in step_calls]:
            return True

    return False


def check_multiverse_math_correctness(sample):
    """Check correctness for MultiverseMath."""
    execution_results = sample.get("execution_results", [])
    ground_truth = sample.get("ground_truth")

    if not execution_results or not ground_truth:
        return False

    # Flatten all execution results
    all_results = []
    for result in execution_results:
        if isinstance(result, list):
            all_results.extend(result)
        else:
            all_results.append(result)

    if not all_results:
        return False

    # Check if any result is an error
    for result in all_results:
        if isinstance(result, str) and "Error" in result:
            return False

    # ground_truth should be a list of expected values
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]

    # Check if all ground_truth values are present in the results
    try:
        present = []
        for gt_val in ground_truth:
            found = False
            for result in all_results:
                if isinstance(gt_val, (int, float)) and isinstance(result, (int, float)):
                    if abs(float(result) - float(gt_val)) < 1e-6:
                        found = True
                        break
                else:
                    if str(result).strip() == str(gt_val).strip():
                        found = True
                        break
            present.append(found)

        return all(present)
    except:
        return False


def check_typewriter_correctness(sample):
    """Check correctness for Typewriter benchmarks by concatenating all characters."""
    execution_results = sample.get("execution_results", [])
    ground_truth = sample.get("ground_truth")

    if not execution_results or not ground_truth:
        return False

    # Typewriter tools return individual characters that need to be concatenated
    all_chars = []
    for result in execution_results:
        if isinstance(result, list):
            for r in result:
                if isinstance(r, str) and "Error" not in r:
                    all_chars.append(r)
        elif isinstance(result, str) and "Error" not in result:
            all_chars.append(result)

    if not all_chars:
        return False

    # Concatenate all characters
    final_result = "".join(all_chars)

    # Compare with ground truth
    return final_result.strip() == str(ground_truth).strip()


def check_tmi_hallucination_correctness(sample):
    """Check correctness for TMI Hallucination by verifying forbidden params are not used."""
    import ast

    generation = sample.get("generation", [])
    ground_truth = sample.get("ground_truth", [])

    if not generation:
        return False

    # Flatten generation
    model_calls_list = []
    for turn in generation:
        if isinstance(turn, list):
            model_calls_list.extend(turn)
        else:
            model_calls_list.append(turn)

    if not model_calls_list:
        return False

    # Get the first call
    model_call = model_calls_list[0] if model_calls_list else ""
    if not model_call:
        return False

    # ground_truth contains args_to_avoid
    args_to_avoid = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    if not args_to_avoid:
        return True

    # Check each forbidden argument
    for arg_to_avoid in args_to_avoid:
        try:
            parsed_call = ast.parse(model_call.strip()).body[0].value

            if parsed_call.args:
                return False

            # Check if the forbidden argument is present
            try:
                # Handle function call API format (**{dict})
                assert len(parsed_call.keywords) == 1
                assert isinstance(parsed_call.keywords[0], ast.keyword)
                assert isinstance(parsed_call.keywords[0].value, ast.Dict)
                for arg_key in parsed_call.keywords[0].value.keys:
                    if arg_key.value == arg_to_avoid:
                        return False
            except Exception:
                # Handle standard function call format
                for keyword in parsed_call.keywords:
                    if keyword.arg == arg_to_avoid:
                        return False
        except Exception:
            return False

    return True


def check_correctness(sample):
    """Main correctness checker that dispatches to appropriate benchmark checker."""
    benchmark_type = sample.get("benchmark_type", "")

    # Load tools for this benchmark
    tools = load_tools_for_benchmark(benchmark_type)

    # Dispatch to appropriate checker
    if benchmark_type in ["nvd_library", "virustotal", "it_type0", "it_type1", "ticket_tracking"]:
        return check_single_turn_correctness(sample, tools)
    elif benchmark_type == "tmi_hallucination":
        return check_tmi_hallucination_correctness(sample)
    elif benchmark_type == "climate":
        return check_climate_correctness(sample, tools)
    elif benchmark_type == "cvecpe":
        return check_cvecpe_correctness(sample, tools)
    elif benchmark_type == "virustotal_agentic":
        return check_virustotal_agentic_correctness(sample, tools)
    elif benchmark_type == "langchain_math":
        return check_langchain_math_correctness(sample)
    elif benchmark_type == "langchain_relational":
        return check_langchain_relational_correctness(sample, tools)
    elif benchmark_type in ["langchain_typewriter_hard", "langchain_multitool_typewriter_hard"]:
        return check_typewriter_correctness(sample)
    elif benchmark_type == "multiverse_math_hard":
        return check_multiverse_math_correctness(sample)
    else:
        LOG.warning(f"Unknown benchmark type: {benchmark_type}")
        return False


def eval_nexusbench(eval_config):
    """Main evaluation function for NexusBench."""
    # Support both input_file (single) and input_files (multiple)
    input_files = eval_config.get("input_files", [])
    if not input_files:
        input_file = eval_config.get("input_file")
        if input_file:
            input_files = [input_file]

    if not input_files:
        LOG.error("No input files specified for evaluation")
        return

    for input_file in unroll_files(input_files):
        input_file = Path(input_file)
        LOG.info(f"Evaluating {input_file}")

        # Read samples
        samples = []
        with open(input_file, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Check correctness for each sample
        for sample in samples:
            sample["is_correct"] = check_correctness(sample)

        # Calculate metrics
        total = len(samples)
        correct = sum(1 for s in samples if s.get("is_correct", False))
        accuracy = correct / total if total > 0 else 0.0

        LOG.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

        # Write back samples with is_correct field
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Write metrics
        metrics_file = input_file.parent / "metrics.json"
        benchmark_name = f"nexusbench.{input_file.parent.name}"
        metrics = {
            benchmark_name: {
                "pass@1": {
                    "accuracy": accuracy,
                    "num_entries": total,
                }
            }
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        LOG.info(f"Metrics saved to {metrics_file}")
