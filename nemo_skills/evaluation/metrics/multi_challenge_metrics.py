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

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class MultiChallengeMetrics(BaseMetrics):
    """Metrics for MultiChallenge benchmark.

    MultiChallenge uses an LLM judge to evaluate conversational AI models.
    Each prediction contains 'is_correct' field set by the evaluator.
    """

    def __init__(self):
        super().__init__()
        # Track per-axis metrics
        self.axis_metrics = defaultdict(lambda: defaultdict(float))
        self.axis_totals = defaultdict(int)

    def reset(self):
        super().reset()
        self.axis_metrics = defaultdict(lambda: defaultdict(float))
        self.axis_totals = defaultdict(int)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Get correctness score from the prediction.

        The 'is_correct' field is set by the MultiChallenge evaluator after LLM judge evaluation.
        """
        correctness_dict = {}

        if "is_correct" in prediction:
            correctness_dict["is_correct"] = prediction["is_correct"]
        else:
            # If not evaluated yet, mark as incorrect
            correctness_dict["is_correct"] = False

        return correctness_dict

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        """Return a prediction that evaluates as incorrect."""
        prediction = prediction.copy()
        prediction["is_correct"] = False
        return prediction

    def _update_axis_metrics(self, prediction: dict, score_dict: dict):
        """Update per-axis metrics if axis information is available."""
        axis = prediction.get("axis", "unknown")
        self.axis_totals[axis] += 1

        for score_method, is_correct in score_dict.items():
            if is_correct:
                self.axis_metrics[axis][score_method] += 1

    def update(self, predictions):
        """Update the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                Each prediction should contain 'is_correct' from the evaluator.
        """
        super().update(predictions)

        # Update axis metrics using the first prediction
        if predictions:
            score_dict = self._get_score_dict(predictions[0])
            self._update_axis_metrics(predictions[0], score_dict)

        # Compute standard pass@k and majority@k metrics
        predicted_answers = [pred.get("generation") for pred in predictions]

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get all computed metrics including per-axis breakdown."""
        metrics_dict = super().get_metrics()

        # Add per-axis metrics to the main evaluation mode
        if self.axis_totals:
            for eval_mode in metrics_dict:
                if eval_mode == f"pass@1[avg-of-{self.max_k}]" or (
                    self.max_k <= 1 and eval_mode == "pass@1"
                ):
                    for axis, total in self.axis_totals.items():
                        for score_method, correct_count in self.axis_metrics[axis].items():
                            accuracy = 100.0 * correct_count / total
                            metrics_dict[eval_mode][f"{axis}_accuracy"] = accuracy

        return metrics_dict

    def evaluations_to_print(self):
        """Return which evaluations should be printed in the summary."""
        if self.max_k > 1:
            return [f"pass@1[avg-of-{self.max_k}]", f"majority@{self.max_k}", f"pass@{self.max_k}"]
        else:
            return ["pass@1"]

    def metrics_to_print(self):
        """Control which metrics are displayed in the summary table."""
        from nemo_skills.evaluation.metrics.base import default_formatting

        return {
            "is_correct": default_formatting,
            "num_entries": default_formatting,
            "avg_tokens": default_formatting,
        }


def compute_metrics(results):
    axis_counts = defaultdict(lambda: {'passed': 0, 'total': 0, 'questions': set()})

    for result in results:
        question_id = result.get('question_id')
        axis = result.get('axis', 'UNKNOWN')
        is_correct = result.get('is_correct', False)

        if question_id not in axis_counts[axis]['questions']:
            axis_counts[axis]['total'] += 1
            axis_counts[axis]['questions'].add(question_id)
            if is_correct:
                axis_counts[axis]['passed'] += 1

    axis_scores = {}
    for axis, counts in axis_counts.items():
        axis_scores[axis] = (counts['passed'] / counts['total'] * 100) if counts['total'] > 0 else 0.0

    overall_score = sum(axis_scores.values()) / len(axis_scores) if axis_scores else 0.0
    total_questions = sum(counts['total'] for counts in axis_counts.values())
    correct_questions = sum(counts['passed'] for counts in axis_counts.values())

    return {
        'overall_score': overall_score,
        'axis_scores': axis_scores,
        'total_questions': total_questions,
        'correct_questions': correct_questions,
        'accuracy': (correct_questions / total_questions * 100) if total_questions > 0 else 0.0,
    }


def format_metrics_report(metrics):
    report = [
        "=" * 60,
        "MultiChallenge Evaluation Results",
        "=" * 60,
        f"\nOverall Score: {metrics['overall_score']:.2f}%",
        f"Total Questions: {metrics['total_questions']}",
        f"Correct Questions: {metrics['correct_questions']}",
        f"Overall Accuracy: {metrics['accuracy']:.2f}%",
        "\n" + "-" * 60,
        "Axis Scores:",
        "-" * 60,
    ]

    for axis, score in sorted(metrics['axis_scores'].items()):
        report.append(f"{axis:30s}: {score:6.2f}%")

    report.append("=" * 60)
    return "\n".join(report)
