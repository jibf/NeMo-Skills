def type_letter(letter: str) -> str:
    """Print the given letter. Accepts only a single letter, like type_writer(letter='a')"""
    return letter


def type_character(character: str) -> str:
    """Print the given character. Accepts only a single character, like type_character(character='@')"""
    return character


type_letter_json = {
    "name": "type_letter",
    "description": """Print the given letter. Accepts only a single letter, like type_writer(letter='a')""",
    "parameters": {
        "type": "object",
        "properties": {
            "letter": {
                "type": "string",
                "description": """Print the given letter. Accepts only a single letter, like type_writer(letter='a')""",
            }
        },
        "required": ["letter"],
    },
}

type_character_json = {
    "name": "type_character",
    "description": """Print the given character. Accepts only a single character, like type_writer(letter='@')""",
    "parameters": {
        "type": "object",
        "properties": {
            "character": {
                "type": "string",
                "description": """Print the given character. Accepts only a single character, like type_character(character='@')""",
            }
        },
        "required": ["character"],
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
