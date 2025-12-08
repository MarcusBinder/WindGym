#!/usr/bin/env python3
"""
Generate API reference documentation using Sphinx.

This script builds Sphinx documentation and converts it to markdown
for integration with Docusaurus.
"""

import os
import sys
import subprocess
import re
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        raise


def install_dependencies():
    """Install required Sphinx dependencies if not already installed."""
    dependencies = [
        'sphinx>=7.0.0',
        'myst-parser>=2.0.0',
        'sphinx-markdown-builder>=0.6.0',
    ]

    print("Checking Sphinx dependencies...")
    for dep in dependencies:
        try:
            import importlib
            module_name = dep.split('>=')[0].replace('-', '_')
            importlib.import_module(module_name)
            print(f"  ✓ {dep.split('>=')[0]} is already installed")
        except ImportError:
            print(f"  Installing {dep}...")
            try:
                run_command(f"{sys.executable} -m pip install '{dep}' --quiet")
                print(f"  ✓ {dep.split('>=')[0]} installed successfully")
            except Exception as e:
                print(f"  ✗ Failed to install {dep}: {e}", file=sys.stderr)
                return False

    return True


def convert_rst_to_markdown(rst_content):
    """
    Convert simple RST to Markdown.
    This handles basic conversions for our use case.
    """
    md = rst_content

    # Convert RST headers to markdown
    md = re.sub(r'^=+$', lambda m: '#' * len(m.group()), md, flags=re.MULTILINE)
    md = re.sub(r'^-+$', lambda m: '##' * len(m.group()), md, flags=re.MULTILINE)

    # Convert code blocks
    md = re.sub(r'\.\. code-block:: python\n\n', '```python\n', md)

    # Convert autoclass directives to a simple format
    md = re.sub(r'\.\. autoclass:: (.*?)\n.*?:members:.*?(?=\n\n|\Z)',
                lambda m: f'### {m.group(1).split(".")[-1]}\n',
                md,
                flags=re.DOTALL)

    return md


def build_sphinx_docs():
    """Build Sphinx documentation."""
    base_dir = Path(__file__).parent.parent
    sphinx_dir = base_dir / "docs_sphinx"
    build_dir = sphinx_dir / "_build"

    print("\nBuilding Sphinx documentation...")

    # Clean previous build
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print("  Cleaned previous build")

    # Try to build with markdown builder first
    try:
        print("  Attempting to build with sphinx-markdown-builder...")
        run_command(
            f"{sys.executable} -m sphinx -b markdown {sphinx_dir} {build_dir}/markdown --keep-going",
            cwd=base_dir
        )

        # Read the generated markdown
        output_file = build_dir / "markdown" / "index.md"
        if output_file.exists():
            with open(output_file, 'r') as f:
                content = f.read()
            print("  ✓ Successfully built with sphinx-markdown-builder")
            return content
    except Exception as e:
        print(f"  sphinx-markdown-builder not available or failed: {e}")

    # Fallback: build HTML and extract text
    try:
        print("  Falling back to HTML build...")
        run_command(
            f"{sys.executable} -m sphinx -b html {sphinx_dir} {build_dir}/html --keep-going",
            cwd=base_dir
        )

        # For now, just use the RST content as a base
        rst_file = sphinx_dir / "index.rst"
        with open(rst_file, 'r') as f:
            rst_content = f.read()

        # Convert to markdown (basic conversion)
        content = convert_rst_to_markdown(rst_content)
        print("  ✓ Built HTML (using RST as base for markdown)")
        return content
    except Exception as e:
        print(f"  ✗ Failed to build Sphinx documentation: {e}", file=sys.stderr)
        raise


def post_process_markdown(content):
    """Post-process the generated markdown for Docusaurus compatibility."""
    # Sphinx already includes the header and footer from index.rst,
    # so we just need to clean up the markdown a bit

    # Replace the RST-style header with markdown
    if content.startswith("WindGym API Reference\n"):
        content = content.replace("WindGym API Reference\n", "# API Reference\n", 1)

    # Escape curly braces for MDX compatibility
    # MDX treats { and } as JSX expression delimiters, so we need to escape them
    # Split by code blocks to avoid escaping inside code
    parts = []
    in_code_block = False
    lines = content.split('\n')

    for line in lines:
        # Check if we're entering/exiting a code block
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            parts.append(line)
        elif in_code_block:
            # Don't escape inside code blocks
            parts.append(line)
        else:
            # Escape curly braces outside code blocks
            # Replace { with \{ and } with \}
            line = line.replace('{', '\\{').replace('}', '\\}')
            parts.append(line)

    content = '\n'.join(parts)

    return content


def main():
    """Main entry point."""
    print("=" * 60)
    print("Generating API reference documentation using Sphinx")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent
    output_path = base_dir / "docusaurus-site" / "docs" / "api-reference.md"

    # Install dependencies
    if not install_dependencies():
        print("\n✗ Failed to install dependencies", file=sys.stderr)
        sys.exit(1)

    try:
        # Build Sphinx docs
        content = build_sphinx_docs()

        # Post-process for Docusaurus
        markdown = post_process_markdown(content)

        # Write output
        output_path.write_text(markdown)

        print(f"\n{'=' * 60}")
        print(f"✓ API reference generated successfully")
        print(f"  Output: {output_path.relative_to(Path.cwd())}")
        print(f"  Size: {len(markdown)} characters")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n✗ Error generating API reference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
