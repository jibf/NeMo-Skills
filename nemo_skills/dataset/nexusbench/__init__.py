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

DATASET_GROUP = "tool"

SPLITS = [
    "nvd_library",
    "virustotal",
    "it_type0",
    "it_type1",
    "ticket_tracking",
    "tmi_hallucination",
    "climate",
    "cvecpe",
    "virustotal_agentic",
    "langchain_relational",
    "langchain_math",
    "multiverse_math_hard",
    "langchain_typewriter_hard",
    "langchain_multitool_typewriter_hard",
]

IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.nexusbench.nexusbench_score"
BENCHMARKS = {f"nexusbench.{split}": {} for split in SPLITS}
