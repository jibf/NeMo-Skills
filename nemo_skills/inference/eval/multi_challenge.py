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
from dataclasses import asdict, field
from pathlib import Path

import hydra

from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MultiChallengeGenerationConfig(GenerateSolutionsConfig):
    """MultiChallenge benchmark generation."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    attempts: int = 1
    prompt_format: str = "openai"
    prompt_config: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_multi_challenge_generation_config", node=MultiChallengeGenerationConfig)


class MultiChallengeGenerationTask(GenerationTask):
    """Generation task for MultiChallenge benchmark."""

    def __init__(self, cfg: MultiChallengeGenerationConfig):
        super().__init__(cfg)
        self.attempts = cfg.attempts

    def log_example_prompt(self, data):
        LOG.info("Example conversation:")
        if data and len(data) > 0:
            conversation = data[0].get("conversation", [])
            for turn in conversation[:2]:
                LOG.info(f"  {turn['role']}: {turn['content'][:100]}...")

    def setup_prompt(self):
        return None

    def preprocess_data(self, data):
        for data_point in data:
            if "conversation" in data_point and "messages" not in data_point:
                data_point["messages"] = data_point["conversation"]
        return data

    async def process_single_datapoint(self, data_point, all_data):
        from dataclasses import asdict as dc_asdict, is_dataclass

        if is_dataclass(self.cfg.inference):
            inference_params = dc_asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

        responses = []
        for attempt_idx in range(self.attempts):
            try:
                prompt = self.fill_prompt(data_point, all_data)
                output_dict = await self.generate_with_semaphore(prompt=prompt, **inference_params)
                responses.append(output_dict.get("generation", ""))
            except Exception as e:
                LOG.error(f"Error in attempt {attempt_idx + 1}: {str(e)}")
                responses.append(f"Error: {str(e)}")

        return {
            "responses": responses,
            "generation": responses[0] if responses else "",
        }


@hydra.main(version_base=None, config_name="base_multi_challenge_generation_config")
def main(cfg: MultiChallengeGenerationConfig):
    cfg = MultiChallengeGenerationConfig(**cfg)
    setup_logging(disable_hydra_logs=False)

    LOG.info("=" * 80)
    LOG.info("MultiChallenge Generation")
    LOG.info(f"Input:  {cfg.input_file}")
    LOG.info(f"Output: {cfg.output_file}")
    LOG.info(f"Attempts: {cfg.attempts}")
    LOG.info("=" * 80)

    if cfg.dry_run:
        LOG.info("Dry run mode - skipping generation")
        return

    task = MultiChallengeGenerationTask(cfg)
    task.generate()

    LOG.info("Generation completed successfully!")
    LOG.info(f"Results saved to: {cfg.output_file}")


if __name__ == "__main__":
    main()
