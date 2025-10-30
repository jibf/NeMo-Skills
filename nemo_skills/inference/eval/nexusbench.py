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

import ast
import importlib
import json
import logging
import sys
from dataclasses import asdict, field

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
class NexusBenchGenerationConfig(GenerateSolutionsConfig):
    """NexusBench benchmark generation config."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)

    prompt_format: str = "openai"
    prompt_config: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_nexusbench_generation_config", node=NexusBenchGenerationConfig)

# Benchmark type to tool module name mapping
TOOL_MODULE_MAPPING = {
    "nvd_library": "nvdlib",
    "it_type0": "it_hard_0",
    "it_type1": "it_hard_1",
    "langchain_typewriter_hard": "typewriter_hard",
    "langchain_multitool_typewriter_hard": "multitool_typewriter_hard",
    "virustotal_agentic": "virustotal_nested",
    "langchain_relational": "relational",
    "multiverse_math_hard": "langchain_math",
}


class NexusBenchGenerationTask(GenerationTask):
    def __init__(self, cfg: NexusBenchGenerationConfig):
        super().__init__(cfg)
        self.tool_registry = {}

    def log_example_prompt(self, data):
        """Log example prompt for the first sample."""
        if data and len(data) > 0:
            LOG.info("Example prompt:")
            LOG.info(f"  Question: {data[0].get('question', '')[:100]}...")
            LOG.info(f"  Benchmark type: {data[0].get('benchmark_type', '')}")
            LOG.info(f"  Single turn: {data[0].get('single_turn', False)}")

    def setup_prompt(self):
        return None

    def _load_tools_module(self, benchmark_type):
        """Dynamically load the tools module for a benchmark type."""
        # TMI loads tools from data
        if benchmark_type == "tmi_hallucination":
            return None

        if benchmark_type not in self.tool_registry:
            try:
                tool_module_name = TOOL_MODULE_MAPPING.get(benchmark_type, benchmark_type)
                module_name = f"nemo_skills.dataset.nexusbench.tools.{tool_module_name}"
                module = importlib.import_module(module_name)
                self.tool_registry[benchmark_type] = module
                LOG.info(f"Loaded tools module for {benchmark_type}")
            except ImportError as e:
                LOG.error(f"Failed to load tools for {benchmark_type}: {e}")
                raise
        return self.tool_registry[benchmark_type]

    def _get_tools_for_benchmark(self, benchmark_type, data_point=None):
        """Get OpenAI-format tools for a benchmark."""
        # TMI loads tools from data_point
        if benchmark_type == "tmi_hallucination":
            if data_point and "json_tools" in data_point:
                json_tools_str = data_point["json_tools"]
                if isinstance(json_tools_str, str):
                    tool_specs = json.loads(json_tools_str)
                else:
                    tool_specs = json_tools_str

                # Convert to OpenAI format
                tools = []
                for tool_spec in tool_specs:
                    tools.append({"type": "function", "function": tool_spec})
                return tools
            else:
                return []

        # Load from tools module
        tools_module = self._load_tools_module(benchmark_type)

        # Get the tool specifications
        if hasattr(tools_module, "get_all_tools_json"):
            tool_specs = tools_module.get_all_tools_json()
        else:
            raise ValueError(f"Tools module for {benchmark_type} missing get_all_tools_json()")

        # Convert to OpenAI format
        tools = []
        for tool_name, tool_spec in tool_specs.items():
            tools.append({"type": "function", "function": tool_spec})

        return tools

    def _parse_function_call(self, call_str):
        """Parse a function call string like 'multiply(a=2, b=3)'."""
        try:
            tree = ast.parse(call_str.strip())
            if not tree.body:
                return None, None

            call_node = tree.body[0].value
            if not isinstance(call_node, ast.Call):
                return None, None

            func_name = call_node.func.id if isinstance(call_node.func, ast.Name) else None
            if not func_name:
                return None, None

            # Extract keyword arguments
            args = {}
            for keyword in call_node.keywords:
                args[keyword.arg] = ast.literal_eval(keyword.value)

            return func_name, args
        except Exception as e:
            LOG.warning(f"Failed to parse function call '{call_str}': {e}")
            return None, None

    def _execute_tool(self, func_name, args, benchmark_type):
        """Execute a single tool call."""
        try:
            tools_module = self._load_tools_module(benchmark_type)

            # If no tools module, cannot execute
            if tools_module is None:
                return f"Error: No tools module for benchmark type {benchmark_type}"

            if not hasattr(tools_module, func_name):
                return f"Error: Function {func_name} not found"

            func = getattr(tools_module, func_name)
            result = func(**args)
            return result
        except Exception as e:
            return f"Error during execution: {str(e)}"

    def _parse_tool_calls_from_response(self, output_dict):
        """Parse tool calls from model response."""
        tool_calls = []

        response = output_dict.get("response")
        if not response:
            return tool_calls

        message = response.choices[0].message if hasattr(response, "choices") else None
        if not message:
            return tool_calls

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                    tool_calls.append({"name": tc.function.name, "arguments": args, "id": tc.id})
                except Exception as e:
                    LOG.warning(f"Failed to parse tool call: {e}")

        return tool_calls

    async def _generate_single_turn(self, data_point):
        """Generate for single-turn benchmark."""
        question = data_point["question"]
        benchmark_type = data_point["benchmark_type"]

        # Get tools
        tools = self._get_tools_for_benchmark(benchmark_type, data_point)

        # Construct prompt
        messages = [{"role": "user", "content": question}]
        prompt_dict = {
            "prompt": messages,
            "tools": tools,
            "tool_choice": "auto",
            "include_response": True,
            **asdict(self.cfg.inference),
        }

        # Generate
        output_dict = await self.generate_with_semaphore(**prompt_dict)

        # Parse tool calls
        tool_calls = self._parse_tool_calls_from_response(output_dict)

        # For single-turn benchmarks, only use the first tool call
        if tool_calls:
            tool_calls = [tool_calls[0]]

        # Execute tools
        execution_results = []
        for tc in tool_calls:
            result = self._execute_tool(tc["name"], tc["arguments"], benchmark_type)
            execution_results.append(result)

        # Format generation output
        if tool_calls:
            generation = [[f"{tool_calls[0]['name']}(**{tool_calls[0]['arguments']})"]]
        else:
            generation = [[]]

        return {
            "generation": generation,
            "execution_results": execution_results,
            "tool_calls": tool_calls,
            "num_generated_tokens": output_dict.get("num_generated_tokens", 0),
        }

    def _build_messages_from_contextual_history(self, query, contextual_history):
        """Build messages list from contextual history."""
        messages = []

        if contextual_history and len(contextual_history) > 0:
            # Check if we need to add previous queries
            all_queries = set(item["previous_query"] for item in contextual_history).union({query})
            add_previous_query = len(all_queries) > 1

            for idx, item in enumerate(contextual_history):
                # Add user query if needed
                if idx == 0 or (
                    add_previous_query and contextual_history[idx - 1]["previous_query"] != item["previous_query"]
                ):
                    messages.append({"role": "user", "content": item["previous_query"]})

                # Add assistant message
                response = item["previous_response"]
                response_message = response.choices[0].message
                messages.append(
                    {
                        "role": "assistant",
                        "refusal": None,
                        "content": response_message.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in (response_message.tool_calls or [])
                        ]
                        if response_message.tool_calls
                        else None,
                    }
                )

                # Add tool result message(s)
                if getattr(response_message, "tool_calls", None):
                    for tc in response_message.tool_calls:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": tc["function"]["name"],
                                "content": json.dumps(item["previous_result"], default=str),
                            }
                        )

            # Add current query if different from last
            if query != contextual_history[-1]["previous_query"]:
                messages.append({"role": "user", "content": query})
        else:
            messages.append({"role": "user", "content": query})

        return messages

    async def _generate_multi_turn(self, data_point):
        """Generate for multi-turn (agent) benchmark."""
        question = data_point["question"]
        benchmark_type = data_point["benchmark_type"]
        max_turns = data_point.get("max_turns", 20)

        # Get tools
        tools = self._get_tools_for_benchmark(benchmark_type, data_point)

        # Use contextual_history
        contextual_history = []

        all_generations = []
        all_results = []
        total_tokens = 0

        for turn_idx in range(max_turns):
            # Build messages from contextual_history
            messages = self._build_messages_from_contextual_history(question, contextual_history)

            # Construct prompt
            prompt_dict = {
                "prompt": messages,
                "tools": tools,
                "tool_choice": "auto",
                "include_response": True,
                **asdict(self.cfg.inference),
            }

            # Generate
            try:
                output_dict = await self.generate_with_semaphore(**prompt_dict)
            except Exception as e:
                LOG.error(f"Generation failed at turn {turn_idx}: {e}")
                break

            raw_response = output_dict["response"]
            total_tokens += output_dict.get("num_generated_tokens", 0)

            # Parse tool calls
            tool_calls = self._parse_tool_calls_from_response(output_dict)

            # Check termination
            if not tool_calls:
                LOG.info(f"No tool calls at turn {turn_idx}, adding END_TOKEN_PREDICTED and terminating")
                all_generations.append(["END_TOKEN_PREDICTED"])
                break

            # For multi-turn, only use the first tool call per turn
            if tool_calls:
                tool_calls = [tool_calls[0]]

            # Execute tool
            result = self._execute_tool(tool_calls[0]["name"], tool_calls[0]["arguments"], benchmark_type)

            # Record this turn
            turn_generation = f"{tool_calls[0]['name']}(**{tool_calls[0]['arguments']})"
            all_generations.append([turn_generation])
            all_results.append([result])

            # Register context
            contextual_history.append(
                {
                    "previous_query": question,
                    "previous_response": raw_response,
                    "previous_call": turn_generation,
                    "previous_result": result,
                }
            )

        return {
            "generation": all_generations,
            "execution_results": all_results,
            "num_turns": len(all_generations),
            "num_generated_tokens": total_tokens,
        }

    async def process_single_datapoint(self, data_point, all_data):
        """Main entry point for processing a single data point."""
        if data_point.get("single_turn", True):
            return await self._generate_single_turn(data_point)
        else:
            return await self._generate_multi_turn(data_point)


GENERATION_TASK_CLASS = NexusBenchGenerationTask


@hydra.main(version_base=None, config_name="base_nexusbench_generation_config")
def main(cfg: NexusBenchGenerationConfig):
    cfg = NexusBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = NexusBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(NexusBenchGenerationConfig)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        main()
