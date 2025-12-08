#!/usr/bin/env python3
"""
Auto-generate API reference documentation from WindGym source code.

This script uses AST parsing to extract docstrings, signatures, and type information
from the WindGym package without requiring dependencies to be installed.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from an AST node."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
        docstring = ast.get_docstring(node)
        return docstring
    return None


def format_arg(arg: ast.arg, defaults: List[Any] = None, default_offset: int = 0) -> str:
    """Format a function argument with its annotation and default value."""
    result = arg.arg

    # Add type annotation
    if arg.annotation:
        result += f": {ast.unparse(arg.annotation)}"

    # Add default value if available
    if defaults and default_offset >= 0:
        try:
            default = defaults[default_offset]
            result += f" = {ast.unparse(default)}"
        except (IndexError, AttributeError):
            pass

    return result


def format_function_signature(node: ast.FunctionDef) -> str:
    """Format function signature from AST node."""
    args_list = []

    # Handle different argument types
    num_defaults = len(node.args.defaults)
    num_args = len(node.args.args)

    for i, arg in enumerate(node.args.args):
        if arg.arg == 'self' or arg.arg == 'cls':
            continue
        # Calculate if this arg has a default
        default_offset = i - (num_args - num_defaults)
        defaults = node.args.defaults if default_offset >= 0 else None
        args_list.append(format_arg(arg, defaults, default_offset))

    # Handle *args
    if node.args.vararg:
        args_list.append(f"*{node.args.vararg.arg}")

    # Handle **kwargs
    if node.args.kwarg:
        args_list.append(f"**{node.args.kwarg.arg}")

    return f"({', '.join(args_list)})"


def parse_class(node: ast.ClassDef, module_path: str) -> Dict:
    """Parse a class definition from AST."""
    class_info = {
        'name': node.name,
        'docstring': extract_docstring(node),
        'methods': [],
        'init_params': []
    }

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            method_name = item.name

            # Skip private methods except __init__
            if method_name.startswith('_') and method_name not in ['__init__', '__call__']:
                continue

            method_info = {
                'name': method_name,
                'signature': format_function_signature(item),
                'docstring': extract_docstring(item)
            }

            if method_name == '__init__':
                # Extract init parameters
                for arg in item.args.args:
                    if arg.arg != 'self':
                        param_info = {
                            'name': arg.arg,
                            'annotation': ast.unparse(arg.annotation) if arg.annotation else None
                        }
                        class_info['init_params'].append(param_info)
            else:
                class_info['methods'].append(method_info)

    return class_info


def parse_function(node: ast.FunctionDef, module_path: str) -> Dict:
    """Parse a function definition from AST."""
    return {
        'name': node.name,
        'signature': format_function_signature(node),
        'docstring': extract_docstring(node),
        'module': module_path
    }


def parse_python_file(file_path: Path, module_path: str) -> Dict:
    """Parse a Python file and extract classes and functions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return {'classes': [], 'functions': []}

    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Only get top-level classes
            if any(node in getattr(item, 'body', []) for item in ast.walk(tree) if isinstance(item, ast.ClassDef)):
                continue
            classes.append(parse_class(node, module_path))
        elif isinstance(node, ast.FunctionDef):
            # Only get top-level functions
            if any(node in getattr(item, 'body', []) for item in ast.walk(tree) if isinstance(item, (ast.ClassDef, ast.FunctionDef))):
                continue
            if not node.name.startswith('_'):
                functions.append(parse_function(node, module_path))

    return {'classes': classes, 'functions': functions}


def clean_docstring(docstring: str, indent: int = 0) -> str:
    """Clean and format docstring for markdown."""
    if not docstring:
        return ""

    lines = docstring.strip().split('\n')
    # Take only the first paragraph (before the Args/Returns section)
    cleaned = []
    for line in lines:
        line = line.strip()
        if line.startswith(('Args:', 'Arguments:', 'Parameters:', 'Returns:', 'Raises:', 'Yields:', 'Note:', 'Notes:', 'Example:', 'Examples:')):
            break
        if line or cleaned:  # Include line if it's not empty or we've already started
            cleaned.append(line)

    result = ' '.join(cleaned).strip()
    return result if result else docstring.split('\n')[0].strip()


def generate_class_markdown(class_info: Dict, module_name: str) -> str:
    """Generate markdown for a class."""
    md = f"### `{class_info['name']}`\n\n"

    # Add docstring
    if class_info['docstring']:
        md += f"{clean_docstring(class_info['docstring'])}\n\n"

    # Add constructor signature
    if class_info['init_params']:
        md += "```python\n"
        md += f"from {module_name} import {class_info['name']}\n\n"
        md += f"{class_info['name']}(\n"

        for i, param in enumerate(class_info['init_params']):
            line = f"    {param['name']}"
            if param['annotation']:
                # Clean up annotation
                annotation = param['annotation'].replace('typing.', '')
                line += f": {annotation}"

            if i < len(class_info['init_params']) - 1:
                line += ","

            md += f"{line}\n"

        md += ")\n```\n\n"

    # Add methods
    if class_info['methods']:
        md += "**Key Methods:**\n\n"
        for method in class_info['methods']:
            first_line = ""
            if method['docstring']:
                first_line = clean_docstring(method['docstring'])
                if first_line:
                    first_line = f": {first_line}"

            md += f"- `{method['name']}{method['signature']}`{first_line}\n"
        md += "\n"

    md += "---\n\n"
    return md


def generate_api_reference() -> str:
    """Generate complete API reference markdown."""
    base_path = Path(__file__).parent.parent / "WindGym"

    md = "# API Reference\n\n"
    md += "This page provides an auto-generated reference for the main classes and functions in WindGym.\n\n"
    md += "---\n\n"

    # Environment Classes
    md += "## Environment Classes\n\n"

    # WindFarmEnv
    wind_farm_env_path = base_path / "wind_farm_env.py"
    if wind_farm_env_path.exists():
        parsed = parse_python_file(wind_farm_env_path, "WindGym")
        for cls in parsed['classes']:
            if cls['name'] == 'WindFarmEnv':
                md += generate_class_markdown(cls, "WindGym")

    # FarmEval
    farm_eval_path = base_path / "farm_eval.py"
    if farm_eval_path.exists():
        parsed = parse_python_file(farm_eval_path, "WindGym")
        for cls in parsed['classes']:
            if cls['name'] == 'FarmEval':
                md += generate_class_markdown(cls, "WindGym")

    # WindFarmEnvMulti
    wind_env_multi_path = base_path / "wind_env_multi.py"
    if wind_env_multi_path.exists():
        parsed = parse_python_file(wind_env_multi_path, "WindGym")
        for cls in parsed['classes']:
            if cls['name'] == 'WindFarmEnvMulti':
                md += generate_class_markdown(cls, "WindGym")

    # Wrappers
    md += "## Wrappers\n\n"

    wrappers_path = base_path / "wrappers"
    if wrappers_path.exists():
        for wrapper_file in wrappers_path.glob("*.py"):
            if wrapper_file.name == "__init__.py":
                continue
            parsed = parse_python_file(wrapper_file, "WindGym.wrappers")
            for cls in parsed['classes']:
                if cls['name'] in ['NoisyWindFarmEnv', 'RecordEpisodeVals', 'CurriculumWrapper', 'RecordEpisodeStatistics']:
                    md += generate_class_markdown(cls, "WindGym.wrappers")

    # Check core for NoisyWindFarmEnv
    core_path = base_path / "core"
    if core_path.exists():
        for core_file in core_path.glob("*.py"):
            parsed = parse_python_file(core_file, "WindGym.core")
            for cls in parsed['classes']:
                if cls['name'] == 'NoisyWindFarmEnv':
                    md += generate_class_markdown(cls, "WindGym.core")

    # Agent Classes
    md += "## Agent Classes\n\n"

    agents_path = base_path / "Agents"
    if agents_path.exists():
        # Define order of agents
        agent_order = ['BaseAgent', 'PyWakeAgent', 'NoisyPyWakeAgent', 'GreedyAgent', 'RandomAgent', 'ConstantAgent']

        for agent_name in agent_order:
            for agent_file in agents_path.glob("*.py"):
                if agent_file.name == "__init__.py":
                    continue
                parsed = parse_python_file(agent_file, "WindGym.Agents")
                for cls in parsed['classes']:
                    if cls['name'] == agent_name:
                        md += generate_class_markdown(cls, "WindGym.Agents")

    # Noise Models
    md += "## Noise Models\n\n"

    if core_path.exists():
        noise_models = ['WhiteNoiseModel', 'EpisodicBiasNoiseModel', 'HybridNoiseModel', 'MeasurementManager']

        for core_file in core_path.glob("*.py"):
            parsed = parse_python_file(core_file, "WindGym.core")
            for cls in parsed['classes']:
                if cls['name'] in noise_models:
                    md += generate_class_markdown(cls, "WindGym.core")

    # Evaluation Tools
    md += "## Evaluation Tools\n\n"

    utils_path = base_path / "utils"
    if utils_path.exists():
        evaluate_ppo_path = utils_path / "evaluate_PPO.py"
        if evaluate_ppo_path.exists():
            parsed = parse_python_file(evaluate_ppo_path, "WindGym.utils.evaluate_PPO")
            for cls in parsed['classes']:
                if cls['name'] == 'Coliseum':
                    md += generate_class_markdown(cls, "WindGym.utils.evaluate_PPO")

    # Utility Functions
    md += "## Utility Functions\n\n"

    if utils_path.exists():
        gen_layouts_path = utils_path / "generate_layouts.py"
        if gen_layouts_path.exists():
            md += "### Layout Generation\n\n"
            md += "Generate wind farm turbine layouts.\n\n"
            parsed = parse_python_file(gen_layouts_path, "WindGym.utils.generate_layouts")

            for func in parsed['functions']:
                if func['name'] in ['grid_layout', 'circular_layout']:
                    doc = clean_docstring(func['docstring']) if func['docstring'] else ""
                    md += f"**`{func['name']}{func['signature']}`**\n\n"
                    if doc:
                        md += f"{doc}\n\n"

            md += "---\n\n"

    # Related Pages
    md += "## Related Pages\n\n"
    md += "- [Core Concepts](concepts.md) - Detailed explanations of key concepts\n"
    md += "- [Agents](agents.md) - Agent development guide\n"
    md += "- [Simulations](simulations.md) - Running simulations\n"
    md += "- [Evaluations](evaluations.md) - Evaluation tools and methods\n"

    return md


def main():
    """Main entry point."""
    print("Generating API reference documentation from source code...")

    try:
        markdown = generate_api_reference()

        # Write to output file
        output_path = Path(__file__).parent.parent / "docusaurus-site" / "docs" / "api-reference.md"
        output_path.write_text(markdown)

        print(f"✓ API reference generated successfully at: {output_path}")
        print(f"  Generated {len(markdown)} characters of documentation")
        print(f"  Output file: {output_path.relative_to(Path.cwd())}")

    except Exception as e:
        print(f"✗ Error generating API reference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
