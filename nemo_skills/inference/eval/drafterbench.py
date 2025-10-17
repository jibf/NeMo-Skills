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

"""DrafterBench benchmark generation script."""

import json
import logging
import re
import sys
from dataclasses import field
from pathlib import Path

import hydra

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class DrafterBenchGenerationConfig(GenerateSolutionsConfig):
    """DrafterBench benchmark generation configuration."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_config is not None:
            raise ValueError("prompt_config must be None for DrafterBench")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_drafterbench_generation_config", node=DrafterBenchGenerationConfig)


class DrafterBenchGenerationTask(GenerationTask):
    """Task for generating DrafterBench responses."""

    def __init__(self, cfg: DrafterBenchGenerationConfig):
        super().__init__(cfg)

    def setup_prompt(self):
        """DrafterBench uses its own prompt system, not NeMo-Skills prompts."""
        return None

    def log_example_prompt(self, data):
        """Log example prompt for DrafterBench.

        DrafterBench has long system prompts embedded in data, so we just log a summary.
        """
        if data:
            example = data[0]
            system_prompt = example.get("system_prompt", "")
            instruction = example.get("instruction", "")
            LOG.info(
                f"Example DrafterBench prompt:\n"
                f"Task: {example.get('task_type', 'unknown')}\n"
                f"System prompt length: {len(system_prompt)} chars\n"
                f"Instruction: {instruction[:200]}..."
            )

    async def process_single_datapoint(self, data_point: dict, all_data: list[dict]):
        """Process a single DrafterBench data point.

        Args:
            data_point: Dictionary containing:
                - instruction: User instruction
                - system_prompt: System prompt embedded in data
                - groundtruth: Ground truth code
                - task_type: Type of task
                - Other metadata fields

        Returns:
            Dictionary with generated code and metadata
        """
        # Get system prompt from data point (embedded during data preparation)
        system_prompt = data_point.get("system_prompt", "")

        if not system_prompt:
            LOG.warning(f"No system prompt found in data point {data_point.get('id')}")
            return {"generation": "", "error": "missing_system_prompt"}

        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_point["instruction"]},
        ]

        # Generate response
        try:
            # Use 'prompt' parameter for OpenAI-style APIs
            output = await self.llm.generate_async(
                prompt=messages,
                include_response=True,
                temperature=self.cfg.inference.temperature,
                top_p=self.cfg.inference.top_p,
                tokens_to_generate=self.cfg.inference.tokens_to_generate,
                random_seed=self.cfg.inference.random_seed,
            )

            # Extract response text from ModelResponse object (like BFCL does)
            if "response" in output:
                # Server-side parsing: extract from response object
                response_message = output["response"].choices[0].message
                response_text = response_message.content or ""
            else:
                # Fallback to direct generation field
                response_text = output.get("generation", "")

            # Extract Python code from markdown code blocks
            code_pattern = r"```python\s*([^`]+)```"
            code_match = re.search(code_pattern, response_text, re.DOTALL)

            if code_match:
                generated_code = code_match.group(1).strip()
            else:
                # If no markdown code block, try to extract code directly
                generated_code = response_text.strip()

            # Return only JSON-serializable fields (not the ModelResponse object)
            return {
                "generation": generated_code,
                "full_response": response_text,
                "num_generated_tokens": output.get("num_generated_tokens", 0),
            }

        except Exception as e:
            LOG.error(f"Error generating response: {e}")
            return {
                "generation": "",
                "error": str(e),
            }


@hydra.main(version_base=None, config_name="base_drafterbench_generation_config")
def drafterbench_generation(cfg: DrafterBenchGenerationConfig):
    cfg = DrafterBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info(f"Config used: {cfg}")

    task = DrafterBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    DrafterBenchGenerationConfig,
    server_params=server_params,
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        drafterbench_generation()
