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

def compute_score(metrics: dict) -> dict:
    """Aggregate metrics across all MultiChallenge task types."""
    task_type_data = {}
    total_questions = 0

    for key, value in metrics.items():
        if key.startswith("multi_challenge."):
            task_type = key.split(".", 1)[1]
            if "pass@1" in value:
                accuracy = value["pass@1"].get("accuracy", 0.0)
                num_entries = value["pass@1"].get("num_entries", 0)
                task_type_data[task_type] = {'accuracy': accuracy, 'num_entries': num_entries}
                total_questions += num_entries

    if task_type_data:
        accuracies = [data['accuracy'] for data in task_type_data.values()]
        overall_accuracy_unweighted = sum(accuracies) / len(accuracies)
        weighted_sum = sum(data['accuracy'] * data['num_entries'] for data in task_type_data.values())
        overall_accuracy_weighted = weighted_sum / total_questions if total_questions > 0 else 0.0
    else:
        overall_accuracy_unweighted = 0.0
        overall_accuracy_weighted = 0.0

    result = {
        "pass@1": {
            'overall_accuracy_unweighted': overall_accuracy_unweighted,
            'overall_accuracy_weighted': overall_accuracy_weighted,
            'num_entries': total_questions,
        }
    }

    for task_type, data in task_type_data.items():
        result["pass@1"][f'{task_type}_accuracy'] = data['accuracy']

    _print_summary(task_type_data, overall_accuracy_weighted, overall_accuracy_unweighted, total_questions)
    return result


def _print_summary(axis_data, overall_weighted, overall_unweighted, total):
    print("\n" + "=" * 80)
    print(" MultiChallenge Benchmark Results ".center(80))
    print("=" * 80)

    if axis_data:
        print(f"\n{'Task Type':<35} {'Accuracy':<15}")
        print("-" * 80)

        for axis, data in axis_data.items():
            axis_display = axis.replace('_', ' ').title()
            print(f"{axis_display:<35} {data['accuracy']:>6.2f}%")

        print("-" * 80)

    print(f"{'Overall Score (Unweighted)':<35} {overall_unweighted:>6.2f}%        {total:>5} questions")
    print(f"{'Overall Score (Weighted)':<35} {overall_weighted:>6.2f}%")
    print("=" * 80 + "\n")
