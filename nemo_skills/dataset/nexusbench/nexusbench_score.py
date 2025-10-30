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


# Suite definitions based on NexusBench organization
SUITES = {
    "common_api": ["nvd_library", "virustotal"],
    "instruction_heavy": ["it_type0", "it_type1", "ticket_tracking"],
    "reliability": ["tmi_hallucination"],
    "agentic_common": ["climate", "cvecpe", "virustotal_agentic", "langchain_relational"],
    "agentic_reasoning": [
        "langchain_math",
        "multiverse_math_hard",
        "langchain_typewriter_hard",
        "langchain_multitool_typewriter_hard",
    ],
}


def get_accuracy_dict(metrics, split_name):
    """Extract accuracy dict for a specific split."""
    split_key = f"nexusbench.{split_name}"
    if split_key not in metrics:
        return None

    split_dict = metrics[split_key]
    if "pass@1" in split_dict:
        return split_dict["pass@1"]
    return None


def calculate_suite_accuracy(metrics, split_names):
    """Calculate average accuracy across splits in a suite."""
    accuracies = []
    total_entries = 0

    for split_name in split_names:
        acc_dict = get_accuracy_dict(metrics, split_name)
        if acc_dict:
            accuracies.append(acc_dict["accuracy"])
            total_entries += acc_dict["num_entries"]

    if not accuracies:
        return {"accuracy": 0.0, "num_entries": 0}

    # Unweighted average across splits in a suite
    avg_accuracy = sum(accuracies) / len(accuracies)
    return {"accuracy": avg_accuracy, "num_entries": total_entries}


def compute_score(metrics: dict):
    """Compute aggregated scores for NexusBench."""
    suite_results = {}

    # Calculate accuracy for each suite
    for suite_name, split_names in SUITES.items():
        suite_results[suite_name] = calculate_suite_accuracy(metrics, split_names)

    # Calculate overall accuracy (unweighted average across suites)
    suite_accuracies = [result["accuracy"] for result in suite_results.values() if result["num_entries"] > 0]
    total_entries = sum(result["num_entries"] for result in suite_results.values())

    overall_accuracy = sum(suite_accuracies) / len(suite_accuracies) if suite_accuracies else 0.0

    return {
        "overall_accuracy": {
            "accuracy": overall_accuracy,
            "num_entries": total_entries,
        },
        "suites": suite_results,
    }


def main():
    """Aggregate NexusBench results from a directory containing metrics.json files."""
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m nemo_skills.dataset.nexusbench.nexusbench_score <results_dir>")
        print("\nAggregates all metrics.json files from subdirectories and computes overall scores.")
        print("\nExample:")
        print("  python -m nemo_skills.dataset.nexusbench.nexusbench_score ./nexusbench_results")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Collect all metrics
    all_metrics = {}

    # Find all metrics.json files
    for metrics_file in results_dir.glob("*/metrics.json"):
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                all_metrics.update(metrics)
        except Exception as e:
            print(f"Warning: Failed to load {metrics_file}: {e}", file=sys.stderr)

    if not all_metrics:
        print("Error: No metrics files found", file=sys.stderr)
        sys.exit(1)

    # Compute aggregated scores
    scores = compute_score(all_metrics)

    # Print results
    print("\n" + "=" * 80)
    print("NexusBench Evaluation Results")
    print("=" * 80)

    # Print overall accuracy
    overall = scores["overall_accuracy"]
    print(f"\nOverall Accuracy: {overall['accuracy']:.4f} ({overall['num_entries']} total samples)")

    # Print suite results
    print("\n" + "-" * 80)
    print("Suite Results:")
    print("-" * 80)

    for suite_name, suite_data in scores["suites"].items():
        print(f"\n{suite_name}:")
        print(f"  Accuracy: {suite_data['accuracy']:.4f}")
        print(f"  Samples:  {suite_data['num_entries']}")

        # Show individual task results in this suite
        suite_tasks = SUITES[suite_name]
        print(f"  Tasks:")
        for task in suite_tasks:
            task_key = f"nexusbench.{task}"
            if task_key in all_metrics:
                task_data = all_metrics[task_key]["pass@1"]
                print(f"    - {task:40s} {task_data['accuracy']:.4f} ({task_data['num_entries']} samples)")
            else:
                print(f"    - {task:40s} [NOT FOUND]")

    # Print individual task results
    print("\n" + "-" * 80)
    print("Individual Task Results:")
    print("-" * 80)
    print(f"{'Task':<45} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 80)

    for task_key in sorted(all_metrics.keys()):
        task_name = task_key.replace("nexusbench.", "")
        task_data = all_metrics[task_key]["pass@1"]
        print(f"{task_name:<45} {task_data['accuracy']:>10.4f} {task_data['num_entries']:>10}")

    print("=" * 80 + "\n")

    # Save aggregated results
    output_file = results_dir / "aggregated_scores.json"
    with open(output_file, "w") as f:
        json.dump({"scores": scores, "all_metrics": all_metrics}, f, indent=2)

    print(f"Aggregated results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
