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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from nemo_skills.evaluation.metrics.multi_challenge_metrics import (
    compute_metrics,
    format_metrics_report,
)
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MultiChallengeEvaluatorConfig:
    judge_model: str = "openai/gpt-4o-20240806"
    judge_base_url: str = "https://api.openai.com/v1"
    judge_api_key: str | None = None
    max_workers: int = 16
    timeout: int = 300


JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''


def evaluate_single_response(judge_client, response, target_question, pass_criteria, eval_config, item_id):
    try:
        judge_prompt = JUDGE_PROMPT.format(response, target_question)
        judge_response = judge_client.chat.completions.create(
            model=eval_config.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            timeout=eval_config.timeout,
        )
        judge_text = judge_response.choices[0].message.content
        verdict = "YES" if "YES" in judge_text.upper().split()[-10:] else "NO"
        passed = verdict == pass_criteria

        return {
            "reasoning": judge_text,
            "verdict": verdict,
            "passed": passed,
            "item_id": item_id,
            "success": True,
        }
    except Exception as e:
        LOG.error(f"Error evaluating item {item_id}: {str(e)}")
        return {
            "reasoning": f"Error during evaluation: {str(e)}",
            "verdict": "NO",
            "passed": False,
            "item_id": item_id,
            "success": False,
        }


def eval_multi_challenge(cfg):
    eval_config = MultiChallengeEvaluatorConfig(**cfg.eval_config)

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai package: pip install openai")

    judge_client = OpenAI(
        api_key=eval_config.judge_api_key,
        base_url=eval_config.judge_base_url,
    )

    LOG.info(f"Judge model: {eval_config.judge_model}, Workers: {eval_config.max_workers}")

    for jsonl_file in unroll_files(cfg.input_files):
        LOG.info(f"Evaluating {jsonl_file}")

        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        eval_tasks = []
        for idx, item in enumerate(data):
            responses = item.get("responses", [item.get("generation", "")])
            if not isinstance(responses, list):
                responses = [responses]

            for resp_idx, response in enumerate(responses):
                eval_tasks.append({
                    "data_idx": idx,
                    "resp_idx": resp_idx,
                    "response": response,
                    "target_question": item["target_question"],
                    "pass_criteria": item["pass_criteria"],
                    "item_id": f"{item.get('question_id', idx)}_attempt{resp_idx}",
                })

        LOG.info(f"Loaded {len(data)} conversations, {len(eval_tasks)} evaluation tasks")

        results = {}
        with ThreadPoolExecutor(max_workers=eval_config.max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_single_response,
                    judge_client,
                    task["response"],
                    task["target_question"],
                    task["pass_criteria"],
                    eval_config,
                    task["item_id"],
                ): task
                for task in eval_tasks
            }

            with tqdm(total=len(eval_tasks), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    results[(task["data_idx"], task["resp_idx"])] = future.result()
                    pbar.update(1)
        for idx, item in enumerate(data):
            responses = item.get("responses", [item.get("generation", "")])
            if not isinstance(responses, list):
                responses = [responses]

            evaluations = []
            for resp_idx in range(len(responses)):
                key = (idx, resp_idx)
                if key in results:
                    eval_result = results[key]
                    evaluations.append({
                        "reasoning": eval_result["reasoning"],
                        "verdict": eval_result["verdict"],
                        "passed": eval_result["passed"],
                    })
                else:
                    evaluations.append({"reasoning": "Evaluation missing", "verdict": "NO", "passed": False})

            item["evaluations"] = evaluations
            item["is_correct"] = any(e["passed"] for e in evaluations)
            item["num_passed"] = sum(e["passed"] for e in evaluations)
            item["num_total"] = len(evaluations)

        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        metrics = compute_metrics(data)
        report = format_metrics_report(metrics)
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60 + "\n")

        output_dir = Path(jsonl_file).parent
        task_type = output_dir.name

        nested_metrics = {
            f"multi_challenge.{task_type}": {
                "pass@1": {
                    "accuracy": metrics['overall_score'],
                    "num_entries": metrics['total_questions'],
                }
            }
        }

        with open(output_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(nested_metrics, f, indent=2)
        with open(output_dir / "report.txt", 'w', encoding='utf-8') as f:
            f.write(report)

        LOG.info(f"Evaluation complete. Metrics saved to {output_dir / 'metrics.json'}")
