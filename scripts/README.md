# API Documentation Generation

This directory contains scripts for auto-generating API reference documentation from the WindGym source code.

## Quick Start

To regenerate the API reference documentation:

```bash
python scripts/generate_api_docs.py
```

This will:
1. Parse the WindGym source code using AST (Abstract Syntax Tree)
2. Extract docstrings, class signatures, and method information
3. Generate markdown documentation at `docusaurus-site/docs/api-reference.md`

## How It Works

The `generate_api_docs.py` script:

- **No dependencies required**: Uses Python's built-in `ast` module to parse source code
- **Extracts documentation from**:
  - Class docstrings
  - Method signatures and docstrings
  - Type annotations
  - Default parameter values
- **Generates markdown** compatible with Docusaurus

## Automated Regeneration

### Option 1: Manual Regeneration

Run the script whenever you make changes to docstrings:

```bash
python scripts/generate_api_docs.py
```

### Option 2: Git Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python scripts/generate_api_docs.py
git add docusaurus-site/docs/api-reference.md
```

### Option 3: CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Generate API docs
  run: python scripts/generate_api_docs.py
```

## Customization

To add new classes or modify documentation structure, edit `generate_api_docs.py`:

1. Find the relevant section (e.g., "Agent Classes", "Wrappers")
2. Add the class name to the appropriate list
3. The script will automatically extract and format the documentation

## Maintenance

The script should be updated when:
- New major classes are added to WindGym
- The module structure changes significantly
- Documentation format requirements change

## Troubleshooting

**Issue**: Missing classes in generated docs
**Solution**: Check that the class is listed in the appropriate section of `generate_api_docs.py`

**Issue**: Malformed signatures
**Solution**: Ensure type annotations in source code use standard Python typing syntax

**Issue**: Missing docstrings
**Solution**: Add docstrings to classes and methods in the source code

## Alternative Tools

If you need more advanced documentation features, consider:

- **Sphinx + autodoc**: Industry standard, very powerful
- **pdoc**: Simpler than Sphinx, good for smaller projects
- **pydoc-markdown**: Generates markdown from docstrings

The current custom script was chosen for:
- Zero external dependencies
- Simple integration with existing Docusaurus setup
- Full control over output format
