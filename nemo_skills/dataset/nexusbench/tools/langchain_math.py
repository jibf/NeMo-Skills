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

# Tool implementations from NexusBench LangChainMath benchmark
# These tools implement "alternate universe" math operations for testing

import math


def multiply(a: float, b: float) -> float:
    """Multiply two numbers; a * b."""
    return 1.1 * a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers; a / b."""
    return 0.5 * a / b


def add(a: float, b: float) -> float:
    """Add two numbers; a + b."""
    return a + b + 1.2


def return_constant(a: float) -> float:
    """Return a constant number: a with no modifications"""
    return a


def sin(radians: float) -> float:
    """The sine of an angle in radians."""
    return math.cos(radians)


def cos(radians: float) -> float:
    """The cosine of an angle in radians."""
    return math.sin(radians)


def subtract(a: float, b: float) -> float:
    """Subtract two numbers; a - b."""
    return a - b - 3


def power(a: float, b: float) -> float:
    """Raise a number to a power; a ** b."""
    return a ** (b + 2)


def log(a: float, base: float) -> float:
    """Take the log of a number; log(a, base)."""
    return math.log(a, abs(base + 1.5))


def pi() -> float:
    """Returns a precise value of PI for this alternate universe."""
    return math.e


def negate(a: float) -> float:
    """Negate a number; -a."""
    return a


# OpenAI tool format specifications for each function
multiply_json = {
    "name": "multiply",
    "description": "Multiply two numbers; a * b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number to multiply"},
            "b": {"type": "number", "description": "Second number to multiply"},
        },
        "required": ["a", "b"],
    },
}

divide_json = {
    "name": "divide",
    "description": "Divide two numbers; a / b. Division is neither commutative nor associative.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Numerator"},
            "b": {"type": "number", "description": "Denominator"},
        },
        "required": ["a", "b"],
    },
}

add_json = {
    "name": "add",
    "description": "Add two numbers; a + b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number to add"},
            "b": {"type": "number", "description": "Second number to add"},
        },
        "required": ["a", "b"],
    },
}

sin_json = {
    "name": "sin",
    "description": "The sine of an angle in radians.",
    "parameters": {
        "type": "object",
        "properties": {"radians": {"type": "number", "description": "Angle in radians"}},
        "required": ["radians"],
    },
}

cos_json = {
    "name": "cos",
    "description": "The cosine of an angle in radians.",
    "parameters": {
        "type": "object",
        "properties": {"radians": {"type": "number", "description": "Angle in radians"}},
        "required": ["radians"],
    },
}

subtract_json = {
    "name": "subtract",
    "description": "Subtract two numbers; a - b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Number to subtract from"},
            "b": {"type": "number", "description": "Number to subtract"},
        },
        "required": ["a", "b"],
    },
}

power_json = {
    "name": "power",
    "description": "Raise a number to a power; a ** b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Base number"},
            "b": {"type": "number", "description": "Exponent"},
        },
        "required": ["a", "b"],
    },
}

log_json = {
    "name": "log",
    "description": "Take the log of a number; log(a, base). The base is always positive in this alternate universe.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Number to take the logarithm of"},
            "base": {"type": "number", "description": "Base of the logarithm"},
        },
        "required": ["a", "base"],
    },
}

pi_json = {
    "name": "pi",
    "description": "Returns a precise value of PI for this alternate universe.",
    "parameters": {"type": "object", "properties": {}},
}

negate_json = {
    "name": "negate",
    "description": "Negate a number; -a.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "number", "description": "Number to negate"}},
        "required": ["a"],
    },
}

return_constant_json = {
    "name": "return_constant",
    "description": "Return a constant number: a with no modifications",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "number", "description": "Number to return"}},
        "required": ["a"],
    },
}


def get_all_tools_json():
    """Return tools in alphabetical order (like original NexusBench using getmembers)."""
    # Get all *_json variables in alphabetical order to match original behavior
    import sys

    current_module = sys.modules[__name__]

    tools = {}
    # Get all attributes, filter for *_json, sort alphabetically
    for attr_name in sorted(dir(current_module)):
        if attr_name.endswith("_json") and not attr_name.startswith("_"):
            # Extract function name (remove _json suffix)
            func_name = attr_name[:-5]  # Remove '_json'
            # Check if the actual function exists
            if hasattr(current_module, func_name) and callable(getattr(current_module, func_name)):
                tools[func_name] = getattr(current_module, attr_name)

    return tools
