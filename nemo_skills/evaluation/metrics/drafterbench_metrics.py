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

"""Metrics for DrafterBench benchmark."""

import logging
import numpy as np

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class DrafterBenchMetrics(BaseMetrics):
    """Metrics calculator for DrafterBench."""

    def __init__(self):
        super().__init__()

    def update(self, sample: dict) -> None:
        """Update metrics with a sample's scores.

        Args:
            sample: Dictionary containing Task_score and other evaluation results
        """
        if "Task_score" not in sample:
            LOG.warning("Sample missing Task_score field")
            return

        task_score_dict = sample["Task_score"]

        # Store the task score
        if "Task_score" in task_score_dict:
            self._results.setdefault("task_scores", []).append(task_score_dict["Task_score"])

        # Store metadata for dimension-wise scoring
        for key in [
            "Structured/Unstructured",
            "Precise|Vague",
            "Complete|Incomplete",
            "Single|Multiple_objects",
            "Single|Multiple_operations",
        ]:
            if key in sample:
                category = sample[key]
                self._results.setdefault(f"scores_by_{key}", {}).setdefault(category, []).append(
                    task_score_dict["Task_score"]
                )

        # Store detailed metrics
        for metric_key in [
            "Success_arguments_define",
            "Total_arguments_define",
            "Success_variable_transfer",
            "Total_variable_transfer",
            "Success_function_calling",
            "Total_function_calling",
            "Success_single_tool_selection",
            "Total_single_tool_selection",
            "Success_multi_tool_selection",
            "Total_multi_tool_selection",
            "Intersected_plan_execution",
            "Total_plans_appeared",
        ]:
            if metric_key in task_score_dict:
                self._results.setdefault(metric_key, []).append(task_score_dict[metric_key])

    def get_metrics(self) -> dict:
        """Calculate and return aggregated metrics.

        Returns:
            Dictionary with average scores across different dimensions
        """
        metrics = {}

        # Overall average task score
        if "task_scores" in self._results:
            metrics["average_task_score"] = float(np.mean(self._results["task_scores"]))

        # Dimension-wise averages
        for dim_key, categories in self._results.items():
            if dim_key.startswith("scores_by_"):
                dim_name = dim_key.replace("scores_by_", "")
                for category, scores in categories.items():
                    metrics[f"{dim_name}_{category}"] = float(np.mean(scores))

        # Detailed metric averages
        for metric_key in [
            "Success_arguments_define",
            "Success_variable_transfer",
            "Success_function_calling",
            "Success_single_tool_selection",
            "Success_multi_tool_selection",
            "Intersected_plan_execution",
        ]:
            if metric_key in self._results:
                total_key = metric_key.replace("Success_", "Total_").replace("Intersected_", "Total_")
                if total_key in self._results:
                    success = sum(self._results[metric_key])
                    total = sum(self._results[total_key])
                    if total > 0:
                        metrics[f"{metric_key}_rate"] = success / total

        return metrics

    @staticmethod
    def get_name() -> str:
        return "drafterbench"
