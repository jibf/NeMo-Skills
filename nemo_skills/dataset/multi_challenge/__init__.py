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

DATASET_GROUP = "chat"

SPLITS = [
    "inference_memory",
    "instruction_retention",
    "reliable_version_editing",
    "self_coherence",
]

IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.multi_challenge.multi_challenge_score"
BENCHMARKS = {f"multi_challenge.{split}": {} for split in SPLITS}

# Evaluation configuration - tells users to add ++eval_type=multi_challenge during generation
GENERATION_ARGS = "++eval_type=multi_challenge"
GENERATION_MODULE = "nemo_skills.inference.eval.multi_challenge"
