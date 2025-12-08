# API Documentation Generation

This directory contains scripts for auto-generating API reference documentation from the WindGym source code using Sphinx.

## Quick Start

To regenerate the API reference documentation:

```bash
python scripts/generate_api_docs_sphinx.py
```

This will:
1. Install Sphinx dependencies if needed (`sphinx`, `myst-parser`, `sphinx-markdown-builder`)
2. Build documentation using Sphinx autodoc
3. Convert to markdown format for Docusaurus
4. Output to `docusaurus-site/docs/api-reference.md`

## How It Works

The documentation system uses **Sphinx** with the following extensions:

- **sphinx.ext.autodoc**: Automatically extract docstrings from Python code
- **sphinx.ext.napoleon**: Support for Google/NumPy-style docstrings
- **sphinx-markdown-builder**: Generate markdown output (instead of HTML)
- **myst-parser**: Enhanced markdown support

### Configuration

Sphinx configuration is in `docs_sphinx/conf.py`:
- Mock imports for unavailable dependencies
- Napoleon settings for docstring parsing
- Autodoc options for class and method documentation

Documentation structure is defined in `docs_sphinx/index.rst`, which lists all classes and modules to document.

## Automated Regeneration

### During Documentation Build

The API documentation is automatically regenerated when building docs:

```bash
cd docusaurus-site
npm run build  # Runs prebuild.sh which generates API docs
```

### Manual Regeneration

Run the script whenever you make changes to docstrings:

```bash
python scripts/generate_api_docs_sphinx.py
```

### Git Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python scripts/generate_api_docs_sphinx.py
git add docusaurus-site/docs/api-reference.md
```

### CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Install docs dependencies
  run: pip install -e ".[docs]"

- name: Generate API docs
  run: python scripts/generate_api_docs_sphinx.py
```

## Installation

### Required Dependencies

Install documentation dependencies with:

```bash
pip install -e ".[docs]"
```

This installs:
- `sphinx>=7.0.0`
- `myst-parser>=2.0.0`
- `sphinx-markdown-builder>=0.6.0`

Alternatively, the script will auto-install dependencies when run.

## Customization

### Adding New Classes

To add new classes to the documentation:

1. Edit `docs_sphinx/index.rst`
2. Add an `autoclass` directive:

```rst
.. autoclass:: WindGym.YourNewClass
   :members:
   :undoc-members:
   :show-inheritance:
```

3. Run the generation script

### Modifying Documentation Format

Edit `docs_sphinx/conf.py` to customize:

- **Autodoc options**: Control what gets documented
- **Napoleon settings**: Configure docstring parsing
- **Mock imports**: Add dependencies that shouldn't be imported

### Changing Output Format

The script generates markdown by default. To generate HTML:

```bash
cd docs_sphinx
sphinx-build -b html . _build/html
```

## Maintenance

Update the documentation when:
- New major classes are added to WindGym
- The module structure changes
- Docstrings are updated or improved

## Troubleshooting

**Issue**: ImportError for missing dependencies
**Solution**: Add the module to `autodoc_mock_imports` in `docs_sphinx/conf.py`

**Issue**: Docstring formatting warnings
**Solution**: Use Google or NumPy style docstrings, or update Napoleon settings

**Issue**: Missing classes in output
**Solution**: Add them to `docs_sphinx/index.rst`

**Issue**: Markdown formatting issues
**Solution**: Check `sphinx-markdown-builder` output or adjust post-processing in the script

## Comparison with Alternatives

### Why Sphinx?

- **Industry standard**: Used by most major Python projects
- **Powerful**: Cross-references, multiple output formats, extensive ecosystem
- **Flexible**: Highly customizable with many extensions
- **Well-documented**: Extensive documentation and community support

### Alternative Tools

- **pdoc**: Simpler, but less powerful and customizable
- **Custom AST parser** (`generate_api_docs.py`): No dependencies, but limited features
- **pydoc-markdown**: Good for GitHub, less suited for Docusaurus

## Documentation Structure

The generated documentation includes:

- **Environment Classes**: `WindFarmEnv`, `FarmEval`, `WindFarmEnvMulti`
- **Wrappers**: `CurriculumWrapper`, `RecordEpisodeVals`, `NoisyWindFarmEnv`
- **Agents**: `BaseAgent`, `PyWakeAgent`, `GreedyAgent`, etc.
- **Noise Models**: `WhiteNoiseModel`, `EpisodicBiasNoiseModel`, etc.
- **Evaluation Tools**: `Coliseum`
- **Utility Functions**: Layout generation, etc.

## Advanced Usage

### Building Different Formats

Sphinx can generate multiple output formats:

```bash
# HTML (for standalone docs)
sphinx-build -b html docs_sphinx docs_sphinx/_build/html

# Markdown (for Docusaurus)
sphinx-build -b markdown docs_sphinx docs_sphinx/_build/markdown

# LaTeX/PDF
sphinx-build -b latex docs_sphinx docs_sphinx/_build/latex
```

### Using Sphinx Directly

For more control, use Sphinx commands directly:

```bash
cd docs_sphinx

# Clean build
sphinx-build -E -b markdown . _build/markdown

# With warnings as errors
sphinx-build -W -b markdown . _build/markdown

# Parallel build
sphinx-build -j auto -b markdown . _build/markdown
```

## Best Practices

1. **Write good docstrings**: Use Google or NumPy style
2. **Keep docs_sphinx/index.rst updated**: Add new classes as they're created
3. **Run generation before commits**: Ensure docs stay in sync with code
4. **Test the build**: Check that Sphinx builds without errors
5. **Review output**: Verify the generated markdown renders correctly in Docusaurus

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [NumPy Style Docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
